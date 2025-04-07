#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mit_bih_afib_dataset.py
@Time    : 2025/03/26 17:04:36
@Desc    : 
"""


import os
import re
import mne
import torch
import wfdb
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class MITBIHAFIBDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './mit-bih-arrhythmia-database-1.0.0',
        pre_offset : int = 100,
        post_offset : int = 200,
        num_channel: int = 62,
        signal_types: list = ['ecg'],
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
        verbose: bool = True,
    ) -> None:
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'pre_offset': pre_offset,
            'post_offset': post_offset,
            'num_channel': num_channel,
            'signal_types': signal_types,
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
        # save all arguments to __dict__
        self.__dict__.update(params)
        super().__init__(**params) 
        
    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        data_files = os.listdir(root_path)
        data_files = np.array(data_files)
        integers = [int(re.search(r'\d+', item).group()) for item in data_files if re.search(r'\d+', item)]
        file_int = set(integers)
        file_int = list(file_int)
        if 256 in file_int:
            file_int.remove(256)
        file_list = [os.path.join(root_path, str(item)) for item in file_int]
        file_list = sorted(file_list)
        print(file_list)
        return file_list

    @staticmethod
    def read_record(record: str, **kwargs):
        data = wfdb.rdsamp(record)
        annotation = wfdb.rdann(record,'atr')
        ecg_signals = data[0].transpose()
        sampling_rate = data[1]['fs']
        ecg_channels = data[1]['sig_name']
        R_location = np.array(annotation.sample)
        labels = np.array(annotation.symbol)
        ecg = {
            'signals': ecg_signals,
            'channels': ecg_channels,
            'sampling_rate': sampling_rate
        }
        return {
            'ecg': ecg,
            'R_location': R_location,
            'labels': labels
        }
       
    @staticmethod
    def process_record(
        record, 
        result,
        signal_types,
        pre_offset,
        post_offset,
        before_trial,
        offline_transform,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        file_name = os.path.splitext(os.path.basename(record))[0]
        if before_trial is not None:
            try:
                for signal_type in signal_types:
                    if signal_type in before_trial:
                        result[signal_type] = before_trial[signal_type](result[signal_type])
            except (KeyError, ValueError) as e:
                print(f'Error in processing record {file_name}: {e}')
                return None
        # print(signal_types)
        # print(pre_offset)
        # print(post_offset)
        aami_mapping = {
            'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # N 类（正常）
            'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',  # S 类（房性心律失常）
            'V': 'V', 'E': 'V',  # V 类（室性心律失常）
            'F': 'F'  # F 类（融合搏动）
        }
        symbols = np.array(['N','L','R','A','a','J','S','V','F','e','j','E'])
        Index = np.isin(result['labels'], symbols)
        labels = np.array(result['labels'])[Index]
        R_location = np.array(result['R_location'])[Index]
        a_data = result['ecg']['signals'].copy()
        data_len = result['ecg']['signals'].shape[1]
        
        for index, item in enumerate(labels):
            # print(index)
            clip_id = f'{index}_{file_name}'
            label = aami_mapping.get(item)
            R_time = R_location[index]
            # print(R_time)
            # print(result['ecg']['signals'].shape)
            if R_time-pre_offset < 0 or R_time+post_offset > data_len:
                # print(f'R_time is out of range')
                continue
            data = a_data[:,R_time-pre_offset:R_time+post_offset]
            info = {
                'clip_id': clip_id,
                'subject_id': file_name,
                'label': aami_mapping[label]
            }
            result = {
                'ecg': {
                    'signals': data,
                    'sampling_rate': result['ecg']['sampling_rate'],
                    'channels': result['ecg']['channels'],
                }
            }
            if not offline_transform is None:
                try:
                    for signal_type in signal_types:
                        if signal_type in offline_transform:
                            result[signal_type] = offline_transform[signal_type](result[signal_type])
                except (KeyError, ValueError) as e:
                    print(f'Error in processing record {file_name}: {e}')
                    return None
            result.update({
                'key': clip_id,
                'info': info
                })
            # print(result)
            yield result
            
    def __getitem__(self, index):
        info = self.read_info(index)
        
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])

        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        label2id = {'N': 0,
                    'S': 1,
                    'V': 2,
                    'F': 3,
                    }
        result['label'] = label2id[info['label']]
        if self.label_transform is not None:
            result['label'] = self.label_transform(result['label'])
        if self.online_transform is not None:
            for signal_type in self.signal_types:
                if signal_type in self.online_transform:
                    result[signal_type] = self.online_transform[signal_type](result[signal_type])
                if 'ToIndexChannels' not in [transform.__class__.__name__ for transform in self.online_transform[signal_type].transforms]:
                    if 'channels' in result[signal_type]:
                        del result[signal_type]['channels']
        else:
            for signal_type in self.signal_types:
                if 'channels' in result[signal_type]:
                    del result[signal_type]['channels']
        return result
