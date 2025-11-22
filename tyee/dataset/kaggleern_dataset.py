#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : kaggleern_dataset.py
@Time    : 2024/12/17 19:48:38
@Desc    : 
"""

import os
import re
import mne
import csv
import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class KaggleERNDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV_2a',
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

    def set_records(self, root_path: str = './KaggleERN/train', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('.csv') and file.startswith('Data'):
                    file_list.append(os.path.join(dirpath, file))
        file_list = sorted(file_list)
        return file_list
    
    def read_record(self, record: str, **kwargs) -> Dict:
        train_id = [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26]
        test_id = [1,3,4,5,8,9,10,15,19,25]
        label_start = [0, 60, 120, 180, 240, 340]
        label_count = 340

        file_name = os.path.splitext(os.path.basename(record))[0][5:]
        subject_id = self.get_subject_id(file_name)
        session_id = self.get_session_id(file_name)
        if subject_id in train_id:
            label_path = os.path.join(os.path.dirname(record), 'TrainLabels.csv')
            subject_index = train_id.index(subject_id)
        elif subject_id in test_id:
            label_path = os.path.join(os.path.dirname(record), "true_labels.csv")
            subject_index = test_id.index(subject_id)
        labels = pd.read_csv(label_path)
        labels = labels.iloc[:, -1:].values
        labels_start = subject_index*label_count + label_start[session_id-1]
        labels_end = subject_index*label_count + label_start[session_id]
        labels = labels[labels_start:labels_end]
        # print(labels)
        raw = pd.read_csv(record)
        data = raw.iloc[:, 1:-2].values
        data = data.T
        channels = ['FP1', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 
                    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 
                    'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 
                    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 
                    'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 
                    'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 
                    'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                    'PO7', 'POZ', 'PO8', 'O1', 'O2']
        eeg = {
            'data': data,
            'freq': 200,
            'channels': channels,
        }
        time = raw['Time'].values        
        feed = raw['FeedBackEvent'].values  
        stim_pos = np.nonzero(feed > 0)[0]  
        segments = []
        for i in range(len(stim_pos)):
            start = time[stim_pos[i]]
            end = time[stim_pos[i]]
            label = int(labels[i][0])
            segments.append({
                'start': start,
                'end': end,
                'value': {
                    'label':{
                        'data': label,
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
                'file_name': file_name,
            }
        }
    
    def get_subject_id(self, file_name) -> str:
        return int(file_name.split('_')[0][1:])
    
    def get_session_id(self, file_name) -> str:
        return int(file_name.split('_')[1][4:])
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]

# from dataset.kaggleern_dataset import KaggleERNDataset
# from dataset.transform import MinMaxNormalize, Offset, Scale, PickChannels

# offline_signal_transform = [
#     MinMaxNormalize(source='eeg', target='eeg'),
#     Offset(offset=-0.5, source='eeg', target='eeg'),
#     Scale(scale_factor=2.0, source='eeg', target='eeg'),
#     PickChannels(channels=['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'], source='eeg', target='eeg')
# ]
# dataset = KaggleERNDataset(
#     root_path='/mnt/ssd/lingyus/KaggleERN/train',
#     io_path='/mnt/ssd/lingyus/tyee_kaggleern/train',
#     io_chunks= 400,
#     io_mode='hdf5',
#     offline_signal_transform=offline_signal_transform,
#     num_worker=8,
# )