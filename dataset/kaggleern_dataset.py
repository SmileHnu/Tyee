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
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict

class KaggleERNDataset(BaseDataset):
    def __init__(self,
                 root_path: str = './KaggleERN/train',
                 label_path: str = './KaggleERN/train_labels.csv',
                 signal_types: list = ['eeg'],
                 offset: int = 0,
                 tmin: int = -0.7,
                 tlen: int = 2,
                 overlap: int = 0,
                 num_channel: int = 22,
                 skip_trial_with_artifacts: bool = False,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True):
        params = {
            'root_path': root_path,
            'label_path': label_path,
            'signal_types': signal_types,
            'offset': offset,
            'tmin': tmin,
            'tlen': tlen,
            'overlap': overlap,
            'num_channel': num_channel,
            'skip_trial_with_artifacts': skip_trial_with_artifacts,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def set_records(self, root_path: str = './KaggleERN/train', label_path: str = './KaggleERN/TrainLabels.csv', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = os.listdir(root_path)
        file_list = [
            os.path.join(root_path, file) for file in file_list
            if file.endswith('.csv')
        ]
        
        return file_list
    
    @staticmethod
    def read_record(record: str, label_path, **kwargs) -> Dict:
        # file_name = os.path.splitext(os.path.basename(record))[0]
        # -- read labels
        labels = []
        with open(os.path.join(label_path), 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if(i>0): labels.append(row)
        labels = dict(labels) # [['S02_Sess01_FB001', '1'],
        raw_data = pd.read_csv(record)
        ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))
        # 将通道处理成大写
        ch_names_kaggle_ern = [x.upper() for x in ch_names_kaggle_ern]
        eeg = {
            'signals': raw_data.values,
            'sampling_rate': 200,
            'ch_names': ch_names_kaggle_ern,
        }
        result = {
            'eeg': eeg,
        }
        # 提取符合record作为前缀的标签
        record_prefix = os.path.splitext(os.path.basename(record))[0]
        record_prefix = record_prefix[5:]
        record_labels = {k: v for k, v in labels.items() if k.startswith(record_prefix)}

        result['labels'] = record_labels
        result['record_prefix'] = record_prefix
        return result
    
    @staticmethod
    def process_record(record: str,
                       result: Dict,
                       signal_types: list,
                       offset: int = 0,
                       tmin: int = -0.7,
                       tlen: int = 2,
                       overlap: int = 0,
                       num_channel: int = 22,
                       skip_trial_with_artifacts: bool = False,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):
        file_name = os.path.splitext(os.path.basename(record))[0]
        # 使用正则表达式提取subject_id和session_id
        match = re.match(r'Data_S(\d+)_Sess(\d+)', file_name)
        if match:
            subject_id = int(match.group(1))
            session_id = int(match.group(2))
            result['subject_id'] = subject_id
            result['session_id'] = session_id
        if not offline_transform is None:
                try:
                    for signal_type in signal_types:
                        if signal_type in offline_transform:
                            result[signal_type] = offline_transform[signal_type](result[signal_type])
                except (KeyError, ValueError) as e:
                    print(f'Error in processing record {file_name}: {e}')
                    return None
                
        raw_data = result['eeg']['signals']
        record_prefix = result['record_prefix']
        data = raw_data[:, 1:-2]
        feed = raw_data[:,-1]
        sample_rate = result['eeg']['sampling_rate']
        feed = torch.tensor(feed)
        stim_pos = torch.nonzero(feed>0)
        # print(stim_pos)
        datas = []
        
        for fb, pos in enumerate(stim_pos, 1):
            clip_id = f'{fb}_{file_name}'
            start_i = max(pos + int(sample_rate * tmin), 0)
            end___i = min(start_i + int(sample_rate * tlen), len(feed))
            print(start_i, end___i)
            trial = data[start_i:end___i, :].T
            record_info = {
                'clip_id': clip_id,
                'session_id': f'{subject_id}_{session_id}',
                'subject_id': subject_id,
                'start_at': start_i,
                'end_at': end___i,
                'signal_type': 'eeg',
                'label': result['labels'][f'{record_prefix}_FB{fb:03d}']
            }
            eeg = {
                'signals': trial,
                'sampling_rate': sample_rate,
                'channels': result['eeg']['ch_names'],
            }
            yield {
                'eeg': eeg,
                'key': clip_id,
                'info': record_info,
            }

    
    def __getitem__(self, index):
        info = self.read_info(index)
        
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])

        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        result['label'] = info['label']
        if self.label_transform is not None:
            result['label'] = self.label_transform(result['label'])
        if self.online_transform is not None:
            for signal_type in self.signal_types:
                if signal_type in self.online_transform:
                    result[signal_type] = self.online_transform[signal_type](result[signal_type])
                if 'ToIndexChannels' not in [transform.__class__.__name__ for transform in self.online_transform[signal_type].transforms]:
                    if 'channels' in result[signal_type]:
                        del result[signal_type]['channels']
        return result
