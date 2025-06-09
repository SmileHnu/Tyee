#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : tuab_dataset.py
@Time    : 2024/11/22 19:18:48
@Desc    : 
"""

import os
import mne
import torch
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator
from dataset.constants.standard_channels import EEG_CHANNELS_ORDER

class TUABDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './tuh_eeg_eabnormal/v3.0.1/edf/train',
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        include_end: bool = False,
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
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
            'io_mode': io_mode,
            'num_worker': num_worker,
            'lazy_threshold': lazy_threshold,
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            if '01_tcp_ar' in dirpath:  
                for file in filenames:
                    if file.endswith('.edf'):
                        file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        return file_list

    def read_record(self, record: str, **kwargs):
        
        Rawdata = mne.io.read_raw_edf(record, preload=True)
        channel_names = Rawdata.ch_names
        new_channel_names = [name.split(' ')[-1].split('-')[0] for name in channel_names]

        mapping = {old_name: new_name for old_name, new_name in zip(channel_names, new_channel_names)}
        Rawdata.rename_channels(mapping)

        drop_channels = [ch for ch in Rawdata.ch_names if ch not in EEG_CHANNELS_ORDER]
        Rawdata.drop_channels(drop_channels)
        
        info = Rawdata.info
        freq = info['sfreq']
        eeg_channels = info['ch_names']
        _, times = Rawdata[:]
        signals = Rawdata.get_data(units='uV')
        
        parts = record.split(os.sep)
        label = parts[-3]
        # print(f'Processing record: {record}, label: {label}')
        Rawdata.close()
        segments = [
            {
                'start': 0,
                'end': times[-1],
                'value':{
                    'label': {
                        'data': label,
                    }
                }
            }
        ]
        eeg = {
            'data': signals,
            'freq': freq,
            'channels': eeg_channels,
        }
        return {
            'signals': {
                'eeg': eeg,
            },
            'labels': {
                'segments': segments,
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(record))[0],
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
                'session_id': self.get_session_id(),
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
        return file_name.split('_')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]


# from dataset.tuab_dataset import TUABDataset
# from dataset.transform.lambd import Lambda
# from dataset.transform.slide_window import SlideWindow
# from dataset.transform import Resample, Compose, Filter, NotchFilter,PickChannels, Mapping
# from dataset.transform.select import Select

# chanels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
# offline_signal_transform = [
#     SlideWindow(window_size=2000,stride=2000,source='eeg', target='eeg'),
# ]
# before_segment_transform = [
#     Compose([
#         PickChannels(channels=chanels),
#         Filter(l_freq=0.1, h_freq=75.0),
#         NotchFilter(freqs=[50.0]),
#         Resample(desired_freq=200),
#         ], source='eeg', target='eeg'
#     ),
# ]
# offline_label_transform = [
#     Mapping(mapping={
#         'abnormal': 1,
#         'normal': 0,
#     }, source='label', target='label'),
#     Select(key='label')
# ]

# dataset = TUABDataset(
#     root_path='/mnt/ssd/lingyus/test',
#     io_path='/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/processed_train',
#     before_segment_transform=before_segment_transform,
#     offline_signal_transform=offline_signal_transform,
#     io_mode='hdf5',
#     num_worker=8
# )