#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : physiop300_dataset.py
@Time    : 2024/12/19 20:24:29
@Desc    : 
"""

import os
import mne
import torch
import numpy as np
import scipy.io as scio
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Generator, Dict, List

class PhysioP300Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './lingyus/erp-based-brain-computer-interface-recordings-1.0.0',
        start_offset: float = -0.7,
        end_offset: float = 1.3,
        include_end: bool = False,
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        lazy_threshold: int = 128,
        verbose: bool = True,
    ) -> None:
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'include_end': include_end,
            'before_segment_transform': before_segment_transform,
            'offline_signal_transform': offline_signal_transform,
            'offline_label_transform': offline_label_transform,
            'online_signal_transform': online_signal_transform,
            'online_label_transform': online_label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_chunks': io_chunks,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'lazy_threshold': lazy_threshold,
            'verbose': verbose
        }
        super().__init__(**params)

    def set_records(self, root_path: str = './erp-based-brain-computer-interface-recordings-1.0.0', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('.edf'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        return file_list
    
    def read_record(self, record: str, **kwargs) -> Dict:
        parts = record.split(os.sep)
        subject_id = parts[-2]  # 's01'
        session_id = os.path.splitext(parts[-1])[0]
        raw = mne.io.read_raw_edf(record)
        raw.rename_channels({ch: ch.upper() for ch in raw.ch_names})
        # print(raw.info)
        events, event_id = mne.events_from_annotations(raw)
        data = raw.get_data(units='uV')
        freq = raw.info['sfreq']
        channels = raw.info['ch_names']
        eeg = {
            'data': data,
            'freq': freq,
            'channels': channels,
        }
        event_map = {}
        tgt = None
        for k,v in event_id.items():
            if k[0:4]=='#Tgt':
                tgt = k[4]
            event_map[v] = k
        # print(event_map)
        assert tgt is not None
        segments = []
        for i in range(len(events)):
            stim = events[i][2]
            t = event_map[stim]
            if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0]=='#':
                continue
            label = 1 if tgt in t else 0
            start = events[i][0] / freq
            end = start
            # if session_id == 'rc01':
            #     print(i, subject_id, session_id)
            segments.append({
                'start': start,
                'end': end,
                'value': {
                    'label':{
                        'data': label
                    }
                }
            })
        
        return {
            'signals':{
                'eeg': eeg
            },
            'labels':{
                'segments': segments,
            },
            'meta':{
                'file_name': f'{subject_id}_{session_id}',
            }
        }
    
    def process_record(
        self,
        signals,
        labels,
        meta,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        signals = self.apply_transform(self.before_segment_transform, signals)
        if signals is None:
            print(f"Skip file {meta['file_name']} due to transform error.")
            return None
        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_label = segment['labels']
            seg_info = segment['info']
            # print(signals['eeg']['data'].shape)
            # print(label['label']['data'])
            segment_id = self.get_segment_id(meta['file_name'], idx)
            seg_signals = self.apply_transform(self.offline_signal_transform, seg_signals)
            seg_label = self.apply_transform(self.offline_label_transform, seg_label)
            if seg_signals is None or seg_label is None:
                print(f"Skip segment {segment_id} due to transform error.")
                continue
            
            seg_info.update({
                'subject_id': self.get_subject_id(meta['file_name']),
                'session_id': self.get_session_id(meta['file_name']),
                'segment_id': self.get_segment_id(meta['file_name'], idx),
                'trial_id': self.get_trial_id(idx),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )

    def get_subject_id(self, file_name) -> str:
        return int(file_name.split('_')[0][1:])
    
    def get_session_id(self, file_name) -> str:
        return int(file_name.split('_')[1][2:])
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]


# from dataset.physiop300_dataset import PhysioP300Dataset
# from dataset.transform import PickChannels, Resample, Filter, Scale, Baseline
# channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
# channels = [ch.upper() for ch in channels]
# offline_signal_transform = [
#     PickChannels(channels=channels, source='eeg', target='eeg'),
#     Baseline(baseline_range=(0, 1435), axis=1, source='eeg', target='eeg'),
#     Filter(l_freq=0, h_freq=120, method= 'iir', source='eeg', target='eeg'),
#     Resample(desired_freq=256, pad="edge", source='eeg', target='eeg'),
#     Scale(scale_factor=1e-3, source='eeg', target='eeg')
# ]
# online_signal_transform = [
#     Baseline(baseline_range=(0, 1434), axis=1, source='eeg', target='eeg'),
# ]
# dataset = PhysioP300Dataset(
#     root_path='/mnt/ssd/lingyus/erp-based-brain-computer-interface-recordings-1.0.0',
#     io_path='/mnt/ssd/lingyus/tyee_physio_p300',
#     io_chunks= 512,
#     io_mode='hdf5',
#     include_end=True,
#     offline_signal_transform=offline_signal_transform,
#     # online_signal_transform=online_signal_transform,
#     num_worker=8,
# )