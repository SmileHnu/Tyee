#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv4_dataset.py
@Time    : 2025/03/28 19:23:58
@Desc    : 
"""

import os
import copy
import scipy.io as scio
from scipy.ndimage import label
from scipy.interpolate import interp1d
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union
from dataset.constants.standard_channels import EEG_CHANNELS_ORDER

class BCICIV4Dataset(BaseDataset):
    def __init__(self,
                 root_path: str = './BCICIV4',
                 chunk_size: int = 40,
                 overlap: int = 0,
                 num_channel: int = 62,
                 signal_types: list = ['ecog'],
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
                 ):
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
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
        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('comp.mat'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        
        return file_list

    @staticmethod
    def read_record(record: str, **kwargs):
        
        data = scio.loadmat(record)
        test_label = scio.loadmat(record.replace('comp.mat','testlabels.mat'))
        train_data = data['train_data'].T
        train_dg = data['train_dg']
        test_data = data['test_data'].T
        test_dg = test_label['test_dg']
        result = {
            'train':{
                'data': train_data,
                'dg': train_dg
            },
            'test':{
                'data': test_data,
                'dg': test_dg
            }
        }
        return result
        
        
    @staticmethod
    def process_record(record, 
                       result,
                       signal_types,
                       chunk_size,
                       overlap,
                       offline_transform,
                       **kwargs):
        file_name = os.path.splitext(os.path.basename(record))[0]
        subject_id = file_name.split('_')[0]
        raw_result = copy.deepcopy(result)
        raw_freq = 1000
        for key in ['train', 'test']:
            # print(f'Processing {file_name}...')
            print(f'Processing record: {file_name}——{key}')
            print(raw_result[key]['data'])
            print(raw_result[key]['data'].shape)
            result = {
                'ecog' : {
                    'signals': raw_result[key]['data'].copy(),
                    'sampling_rate': 1000,
                    'channels': [f"{i}" for i in range(raw_result[key]['data'].shape[0])]
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
                
            new_freq = result['ecog']['sampling_rate']
            # 如果采样率发生变化，重新映射 label
            if new_freq != raw_freq:
                # 计算原始时间轴和新时间轴
                original_time = np.linspace(0, len(raw_result[key]['dg']) / raw_freq, len(raw_result[key]['dg']))
                new_time = np.linspace(0, result['ecog']['signals'].shape[1] / new_freq, result['ecog']['signals'].shape[1])  # 修正为数据的时间轴

                # 初始化插值后的 dg
                interpolated_dg = []

                # 对每个通道分别进行插值
                for channel in range(raw_result[key]['dg'].shape[1]):  # 遍历每个通道
                    channel_data = raw_result[key]['dg'][:, channel]  # 提取当前通道的数据
                    dg_interp = interp1d(original_time, channel_data, kind='linear', fill_value="extrapolate")
                    interpolated_channel = dg_interp(new_time)  # 对当前通道进行插值
                    interpolated_dg.append(interpolated_channel)

                # 将插值后的通道数据重新组合成二维数组
                raw_result[key]['dg'] = np.stack(interpolated_dg, axis=1).astype(raw_result[key]['dg'].dtype)
            dg = raw_result[key]['dg'].copy().T
            ecog = result['ecog']['signals'].copy()
            start = 0
            end = chunk_size
            # 计算重叠步长
            stride = chunk_size - overlap
            idx = 0
            while end <= dg.shape[1]:
                # print(f'Processing trial: {idx}')
                clip_id = f'{idx}_{key}_{file_name}'
                info = {
                    'clip_id': clip_id,
                    'subject_id': subject_id,
                    'trial_id': key,
                    'start_at': start,
                    'end_at': end,
                    'label': dg[:,end-1:end]
                }
                result = {
                    'ecog': {
                        'signals': ecog[:, start:end],
                        'sampling_rate': result['ecog']['sampling_rate'],
                        'channels': result['ecog']['channels'],
                    }
                }

                result.update({
                    'key': clip_id,
                    'info': info
                })
                yield result
                start = start + stride
                end = start + chunk_size
                idx += 1


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
