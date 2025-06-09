#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv2a_dataset.py
@Time    : 2024/12/19 19:40:20
@Desc    : 
"""
import os
import mne
import torch
import numpy as np
import scipy.io as scio
from dataset import BaseDataset
from typing import Any, Callable, Union, Generator, Dict, List

class BCICIV2ADataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV_2a',
        start_offset: float = 2.0,
        end_offset: float = 6.0,
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

    def set_records(self, root_path: str = './BCICIV_2a', **kwargs) -> List:
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        files = os.listdir(root_path)
        gdf_files = []
        mat_files = []

        for file in files:
            if file.endswith('.gdf'):
                gdf_files.append(os.path.join(root_path, file))
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root_path, file))

        gdf_files.sort()
        mat_files.sort()

        record_pairs = []
        for gdf_file, mat_file in zip(gdf_files, mat_files):
            if os.path.splitext(os.path.basename(gdf_file))[0] == os.path.splitext(os.path.basename(mat_file))[0]:
                record_pairs.append([gdf_file, mat_file])

        print(record_pairs)
        return record_pairs
    
    def read_record(self, record: tuple, **kwargs) -> Dict:
        gdf_file, mat_file = record
        # Read the gdf file
        gdf_data = mne.io.read_raw_gdf(gdf_file, preload=True)
        # eog
        eog_gdf_data = gdf_data.copy()
        eog_gdf_data.pick_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        eog_data = eog_gdf_data.get_data(units='uV')
        eog_freq = eog_gdf_data.info['sfreq']
        eog_channels = eog_gdf_data.info['ch_names']
        eog_channels = [ch.split('-')[-1] for ch in eog_channels]
        eog = {
            'data': eog_data,
            'freq': eog_freq,
            'channels': eog_channels,
        }
        # eeg
        gdf_data.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
        eeg_data = gdf_data.get_data(units='uV')
        freq = gdf_data.info['sfreq']
        eeg_channels = gdf_data.info['ch_names']
        eeg_channels = [ch.split('-')[-1] for ch in eeg_channels]
        eeg = {
            'data': eeg_data,
            'freq': freq,
            'channels': eeg_channels,
        }
        # event
        events, event_ids = mne.events_from_annotations(gdf_data)
        start_id = ['768']
        wanted_ids = [event_ids[k] for k in start_id if k in event_ids]
        filtered_events = mne.pick_events(events, include=wanted_ids)
        # Read the mat file
        mat_data = scio.loadmat(mat_file)
        labels = mat_data['classlabel']
        # print(labels)
        # print(len(mat_data['classlabel']))
        segments = []
        for i in range(len(labels)):
            label = labels[i][0] - 1 # 0-left, 1-right, 2-foot, 3-tongue
            # print(label)
            start = filtered_events[i][0]
            end = filtered_events[i][0]
            # print(start, end, filtered_events[i][2])
            segments.append({
                'start': start / freq,
                'end': end / freq,
                'value': {
                    'event': {
                        'data': label,
                    },
                }
            })
        
        return {
            'signals':{
                'eeg': eeg,
                'eog': eog,
            },
            'labels':{
                'segments': segments,
            },
            'meta':{
                'file_name': os.path.splitext(os.path.basename(gdf_file))[0],
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
                'run_id': self.get_run_id(meta['file_name'], idx),
                'trial_id': self.get_trial_id(idx),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )

    def get_subject_id(self, file_name) -> str:
        return file_name[0:3]
    
    def get_session_id(self, file_name) -> str:
        if file_name[3:] == 'T':
            return '0'
        elif file_name[3:] == 'E':
            return '1'
    
    def get_run_id(self, file_name, idx) -> str:
        run_idx = idx % 48
        return f'{run_idx}'
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]
    

# from dataset.bciciv2a_dataset import BCICIV2ADataset
# from dataset.transform import Cheby2Filter, Select

# offline_signal_transform = [
#     Cheby2Filter(l_freq=4, h_freq=40, source='eeg', target='eeg'),
#     Select(key=['eeg']),
# ]
# for i in range(3, 10):
#     dataset = BCICIV2ADataset(
#         root_path=f'/mnt/ssd/lingyus/BCICIV_2a/A0{i}',
#         io_path=f'/mnt/ssd/lingyus/tyee_bciciv2a/A0{i}',
#         io_mode='hdf5',
#         io_chunks=750,
#         offline_signal_transform=offline_signal_transform
#     )
#     print(dataset[0])
#     indices = dataset.info[dataset.info['session_id'] == 0].index.tolist()

#     all_eeg = []
#     for idx in indices:
#         eeg = dataset[idx]['eeg']  
#         all_eeg.append(eeg)

#     all_eeg = np.concatenate(all_eeg, axis=-1) 

#     mean = np.mean(all_eeg)
#     std = np.std(all_eeg)
#     print(f'A0{i}')
#     print('mean:', mean)
#     print('std:', std)

# A01
# mean: -0.004770609852039078
# std: 5.314540361031272
# A02
# mean: 0.0016582006705953179
# std: 5.428212821911444
# A03
# mean: 0.00350088778759517
# std: 6.730600925489113
# A04
# mean: -0.0004827520189063491
# std: 4.942236152492034
# A05
# mean: -0.0002760650782005268
# std: 4.155209908726351
# A06
# mean: -0.004262571623786904
# std: 7.5902684030035585
# A07
# mean: 2.6756102459528735e-05
# std: 4.758070863930625
# A08
# mean: 0.003387182489560944
# std: 9.08903875408772
# A09
# mean: -0.000831605672567208
# std: 9.915488018511994