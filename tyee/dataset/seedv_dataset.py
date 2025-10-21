#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : seedv_dataset.py
@Time    : 2024/12/26 20:12:55
@Desc    : 
"""
import os
import mne
import re
import torch
import pickle
import numpy as np
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class SEEDVFeatureDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './SEED-V',
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
    
        
    def set_records(self, root_path: str = None, **kwargs):
        eeg_path = os.path.join(root_path, 'EEG_DE_features')
        assert os.path.exists(eeg_path), f"EEG feature path {eeg_path} does not exist."
        eog_path = os.path.join(root_path, 'Eye_movement_features')
        assert os.path.exists(eog_path), f"EOG feature path {eog_path} does not exist."
        eeg_file_path_list = os.listdir(eeg_path)
        eeg_file_path_list = [
            os.path.join(eeg_path, file_path) for file_path in eeg_file_path_list
            if file_path.endswith('.npz')
        ]
        eeg_file_path_list = sorted(eeg_file_path_list, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        eog_file_path_list = os.listdir(eog_path)
        eog_file_path_list = [
            os.path.join(eog_path, file_path) for file_path in eog_file_path_list
            if file_path.endswith('.npz')
        ]
        eog_file_path_list = sorted(eog_file_path_list, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
        assert len(eeg_file_path_list) == len(eog_file_path_list), \
            f"Number of EEG files ({len(eeg_file_path_list)}) does not match number of EOG files ({len(eog_file_path_list)})."
        records = []
        for eeg_file, eog_file in zip(eeg_file_path_list, eog_file_path_list):
            if os.path.basename(eeg_file) != os.path.basename(eog_file):
                print(f"Skipping mismatched files: {eeg_file} and {eog_file}")
                continue
            record = (eeg_file, eog_file)
            records.append(record)
        return records
    
    def read_record(self, record: str, **kwargs):
        eeg_file, eog_file = record
        # Load EEG data
        eeg_data_npz = np.load(eeg_file)
        eeg_data_dict = pickle.loads(eeg_data_npz['data'])
        label_dict = pickle.loads(eeg_data_npz['label'])
        # Load EOG data
        eog_data_npz = np.load(eog_file)
        eog_data_dict = pickle.loads(eog_data_npz['data'])
        start = 0
        end = 0 
        segments = []
        eeg_data = []
        eog_data = []
        print(eeg_data_dict.keys())
        for key in eeg_data_dict.keys():
            key_eeg_data = eeg_data_dict[key]
            key_eog_data = eog_data_dict[key]
            key_label = label_dict[key]
            end = start + key_eeg_data.shape[0]
            segments.append({
                'start': start,
                'end': end,
                'value':{
                    'emotion':{
                        'data': key_label,
                    }
                }
            })
            start = end
            eeg_data.append(key_eeg_data)
            eog_data.append(key_eog_data)
        eeg_data = np.concatenate(eeg_data, axis=0)
        eog_data = np.concatenate(eog_data, axis=0)
        return {
            'signals':{
                'eeg':{
                    'data': eeg_data,
                },
                'eog':{
                    'data': eog_data,
                }
            },
            'labels': {
                'segments': segments,
            },
            'meta':{
                'file_name': os.path.splitext(os.path.basename(eeg_file))[0]
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
                'session_id': self.get_session_id(idx),
                'segment_id': self.get_segment_id(meta['file_name'], idx),
                'trial_id': self.get_trial_id(idx),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )
    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        Segment all signal types in signals according to label['segments'], and return signals and labels for each segment.
        The start/end units in label['segments'] are in seconds.
        """
        segments = []
        false = False
        for seg in labels['segments']:
            seg_dict = {'signals': {}, 'labels': seg['value'], 'info': {}}
            start_time = seg['start'] + self.start_offset
            end_time = seg['end'] + self.end_offset
            seg_dict['info']= {
                'start_time': start_time,
                'end_time': end_time,
            }
            for sig_type, sig in signals.items():
                data = sig['data']
                start_idx = int(start_time)
                end_idx = int(end_time)
                if self.include_end:
                    end_idx += 1
                if start_idx < 0 or end_idx > data.shape[0] or start_idx > end_idx:
                    false = True
                    print(f"Invalid segment: {sig_type}, {start_time}, {end_time}, {start_idx}, {end_idx}, {data.shape[0]}")
                    continue
                # print(start_idx, end_idx)
                seg_dict['signals'][sig_type] = {
                    'data': data[start_idx:end_idx, ...],
                }
            if false:
                false = False
                continue
            segments.append(seg_dict)
        return segments
    
    def get_subject_id(self, file_name) -> str:
        return file_name.split('_')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_session_id(self, idx) -> str:
        return str(idx//15)  
    
    def get_trial_id(self, idx) -> str:
        return str(idx % 15)

# from dataset.seedv_dataset import SEEDVFeatureDataset
# from dataset.transform import (
#     Log, SlideWindow, MinMaxNormalize, Concat, Insert, 
#     Squeeze, Select, ToNumpyInt64
# )

# offline_signal_transform = [
#     Log(epsilon=1, source='eog', target='eog'),
#     SlideWindow(window_size=1, stride=1, axis=0, source='eeg', target='eeg'),
#     SlideWindow(window_size=1, stride=1, axis=0, source='eog', target='eog')
# ]

# offline_label_transform = [
#     SlideWindow(window_size=1, stride=1, axis=0, source='emotion', target='emotion')
# ]

# online_signal_transform = [
#     MinMaxNormalize(source='eeg', target='eeg'),
#     MinMaxNormalize(source='eog', target='eog'),
#     Concat(axis=-1, source=["eeg", "eog"], target='eeg_eog'),
#     Insert(
#         indices=[316, 317, 318, 319, 326, 327, 328, 329, 334, 335, 336, 337, 
#                 338, 339, 344, 345, 346, 347, 348, 349, 354, 355, 356, 357, 
#                 358, 359, 369],
#         value=0,
#         axis=-1,
#         source='eeg_eog',
#         target='eeg_eog'
#     ),
#     Squeeze(axis=0, source='eeg_eog', target='eeg_eog'),
#     Select(key=["eeg_eog"])
# ]

# online_label_transform = [
#     Squeeze(axis=0, source='emotion', target='emotion'),
#     ToNumpyInt64(source='emotion', target='emotion')
# ]

# dataset = SEEDVFeatureDataset(
#     root_path='/your/data/path',
#     io_path='/your/io/path',
#     io_chunks=1,
#     io_mode='hdf5',
#     offline_signal_transform=offline_signal_transform,
#     offline_label_transform=offline_label_transform,
#     online_signal_transform=online_signal_transform,
#     online_label_transform=online_label_transform,
#     num_worker=8,
# )