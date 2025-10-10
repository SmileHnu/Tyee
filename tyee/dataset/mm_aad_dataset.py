#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mm_aad_dataset.py
@Time    : 2025/09/30 14:03:32
@Desc    : 
"""

import os
import mne
import torch
import numpy as np
import scipy.io as scio
from dataset import BaseDataset
from typing import Any, Callable, Union, Generator, Dict, List

class MMAADDataset(BaseDataset):
    def __init__(
        self,
        # root_path:
        # EEG-AAD/EEG-AAD_audio_visual/preprocessed/ EEG-AAD/EEG-AAD_audio_visual/raw/
        # EEG-AAD/EEG-AAD_audio_only/preprocessed/ EEG-AAD/EEG-AAD_audio_only/raw/
        # EEG-AAD_audio_only_part2/preprocessed/ EEG-AAD_audio_only_part2/raw/
        root_path: str = '',
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

    def set_records(self, root_path: str = '', **kwargs) -> List:
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        data_path = os.path.join(root_path, 'data')
        label_path = os.path.join(root_path, 'label')
        assert os.path.exists(
            data_path
        ), f'data_path ({data_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        assert os.path.exists(
            label_path
        ), f'label_path ({label_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        
        file_names = os.listdir(data_path)
        file_names = [f for f in file_names if f.endswith('.npy')]
        file_names.sort(key=lambda f: int(f[1:-4]))
        records = []
        for file_name in file_names:
            data_file = os.path.join(data_path, file_name)
            label_file = os.path.join(label_path, file_name)
            assert os.path.exists(
                label_file
            ), f'label_file ({label_file}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
            records.append([data_file, label_file])
        # print(records)
        return records
    
    def read_record(self, record: tuple, **kwargs) -> Dict:
        data_file, label_file = record
        # print(f"Reading record: {data_file}, {label_file}")
        signals = np.load(data_file)  # shape (trials, timepoints, channels)
        labels = np.load(label_file)  # shape (trials, timepoints)
        eeg_channels = ['FP1', 'FP2', 'F11', 'F7', 'F3', 
                        'FZ', 'F4', 'F8', 'F12', 'FT11', 
                        'FC3', 'FCZ', 'FC4', 'FT12', 'T7', 
                        'C3', 'CZ', 'C4', 'T8', 'CP3', 
                        'CPZ', 'CP4', 'M1', 'M2', 'P7', 
                        'P3', 'PZ', 'P4', 'P8', 'O1', 
                        'OZ', 'O2']
        segments = []
        for i in range(len(labels)):
            segment = {
                'start': i,
                'end': i,
                'value': {
                    'label': {
                        'data': labels[i],
                    }
                }
            }
            segments.append(segment)
        return {
            'signals': {
                'eeg': {'data': signals, 'freq': len(signals[0]), 'channels': eeg_channels},
            },
            'labels': {
                'segments': segments
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(data_file))[0]
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
            start_time = seg['start']
            end_time = seg['end']
            seg_dict['info']= {
                'start': start_time,
                'end': end_time,
            }
            for sig_type, sig in signals.items():
                freq = sig['freq']
                data = sig['data']
                start_idx = start_time
                end_idx = end_time
                if self.include_end:
                    end_idx += 1
                if start_idx < 0 or end_idx > data.shape[0] or start_idx > end_idx:
                    false = True
                    print(f"Invalid segment: {sig_type}, {start_time}, {end_time}, {start_idx}, {end_idx}")
                    continue
                # print(start_idx, end_idx)
                seg_dict['signals'][sig_type] = {
                    'data': data[start_idx],
                    'channels': sig.get('channels', []),
                    'freq': freq,
                }
            if false:
                false = False
                continue
            segments.append(seg_dict)
        return segments
    
    def get_subject_id(self, file_name) -> str:
        return file_name[1:]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]