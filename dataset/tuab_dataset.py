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
        root_path: str = './EEG_raw',
        channel_std: str = '01_tcp_ar',
        chunk_size: int = 800,
        overlap: int = 0,
        num_channel: int = 62,
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
            'channel_std': channel_std,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
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
    
    def set_records(self, root_path: str = None, channel_std: str = '01_tcp_ar', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            if channel_std in dirpath:  # 只处理路径中包含 channel_std 的文件
                for file in filenames:
                    if file.endswith('.edf'):
                        file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        return file_list

    @staticmethod
    def read_record(record: str, **kwargs):
        
        Rawdata = mne.io.read_raw_edf(record, preload=True)
        channel_names = Rawdata.ch_names
        new_channel_names = [name.split(' ')[-1].split('-')[0] for name in channel_names]

        mapping = {old_name: new_name for old_name, new_name in zip(channel_names, new_channel_names)}
        Rawdata.rename_channels(mapping)

        drop_channels = [ch for ch in Rawdata.ch_names if ch not in EEG_CHANNELS_ORDER]
        Rawdata.drop_channels(drop_channels)
        
        info = Rawdata.info
        sampling_rate = info['sfreq']
        eeg_channels = info['ch_names']
        signals = Rawdata.get_data(units='uV')
        
        parts = record.split(os.sep)
        label = parts[-3]
        print(f'Processing record: {record}, label: {label}')
        Rawdata.close()

        eeg = {
            'signals': signals,
            'sampling_rate': sampling_rate,
            'channels': eeg_channels,
        }
        result = {
            'eeg': eeg,
            'label': label,
        }
        return result
        
    @staticmethod
    def process_record(
        record, 
        signal_types,
        result,
        offline_transform,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        file_name = os.path.splitext(os.path.basename(record))[0]

        if not offline_transform is None:
            try:
                for signal_type in signal_types:
                    if signal_type in offline_transform:
                        result[signal_type] = offline_transform[signal_type](result[signal_type])
            except (KeyError, ValueError) as e:
                print(f'Error in processing record {file_name}: {e}')
                return None
        
        if result['label'] == 'normal':
            label = 0
        elif result['label'] == 'abnormal':
            label = 1
        signals = result['eeg']['signals'].copy()
        for idx in range(signals.shape[1] // 2000):
            eeg_signal = signals[:, idx * 2000: (idx + 1) * 2000]
            clip_id = f'{idx}_{file_name}'
            subject_id = file_name.split('_')[0]
            result['eeg']['signals'] = eeg_signal
            yield {
                'eeg': result['eeg'],
                'key': clip_id,
                'info': {
                    'signal_types': ['eeg'],
                    'clip_id': clip_id,
                    'subject_id': subject_id,
                    'label': label
                }
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
        else:
            for signal_type in self.signal_types:
                if 'channels' in result[signal_type]:
                    del result[signal_type]['channels']
        return result
