#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : ninapro_db5_dataset.py
@Time    : 2025/03/27 20:41:09
@Desc    : 
"""

import os
import scipy.io as scio
from scipy.ndimage import label
from scipy.interpolate import interp1d
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class NinaproDB5Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './NinaproDB5',
        chunk_size: int = 40,
        overlap: int = 32,
        num_channel: int = 16,
        signal_types: list = ['emg'],
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
                if file.endswith('.mat'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        
        return file_list

    @staticmethod
    def read_record(record: str, **kwargs):
        
        a_data = scio.loadmat(record)
        emg = np.array(a_data['emg']).astype(np.float64).T
        restimulus = np.array(a_data['restimulus'])
        sampling_rate = a_data['frequency'].astype(np.float64)
        glove = np.array(a_data['glove']).astype(np.float64)
        result = {
            'emg': {
                'signals': emg,
                'sampling_rate': sampling_rate,
                'channels': [f"sEMG{i}" for i in range(emg.shape[0])],
            },
            'restimulus': restimulus,
            'glove': glove,
            'freq': sampling_rate,
            
        }
        return result
        
    @staticmethod
    def process_record(
        record, 
        result,
        signal_types,
        chunk_size,
        overlap,
        offline_transform,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        file_name = os.path.splitext(os.path.basename(record))[0]
        subject_id = file_name.split('_')[0]
        session_id = file_name.split('_')[1]
        offset = {
            'E1': 0,
            'E2': 12,
            'E3': 12+17,
        }
        if not offline_transform is None:
            try:
                for signal_type in signal_types:
                    if signal_type in offline_transform:
                        result[signal_type] = offline_transform[signal_type](result[signal_type])
            except (KeyError, ValueError) as e:
                print(f'Error in processing record {file_name}: {e}')
                return None
        restimulus = result['restimulus'].copy()
        glove = result['glove'].copy()
        data = result['emg']['signals'].copy()
        sampling_rate = result['emg']['sampling_rate'].copy()
        freq = result['freq'].copy()
        channels = result['emg']['channels'].copy()
        mask = restimulus > 0
        label_array, num_segments = label(mask)
        # 如果采样率发生变化，重新映射 glove 和 restimulus
        if sampling_rate != freq:
            # 计算原始时间轴和新时间轴
            original_time = np.linspace(0, len(restimulus) / freq, len(restimulus))
            new_time = np.linspace(0, data.shape[1] / sampling_rate, data.shape[1])  # 修正为数据的时间轴


            # 使用插值函数重新映射 restimulus 和 glove
            restimulus_interp = interp1d(original_time, restimulus, kind='nearest', fill_value="extrapolate")
            glove_interp = interp1d(original_time, glove, kind='linear', fill_value="extrapolate")

            # 更新 restimulus 和 glove
            restimulus = restimulus_interp(new_time).astype(restimulus.dtype)
            glove = glove_interp(new_time).astype(glove.dtype)
        
        idx = 0 
        for seg_id in range(1, num_segments + 1):  # 遍历每个手势段

            segment_indices = np.where(label_array == seg_id)[0]  # 该段的索引
            segment_emg = data[:, segment_indices]  # 取出对应的 sEMG 片段（调整索引顺序）
            segment_label = restimulus[segment_indices]  # 取出对应的标签

            unique_labels = np.unique(segment_label)
            if len(unique_labels) > 1:
                raise ValueError(f"Segment {seg_id} 存在多个不同标签: {unique_labels}")

            hand_gesture = unique_labels[0] + offset[session_id] - 1  # 该段唯一的手势标签
            
            # 滑动窗口提取样本
            # 计算重叠步长
            stride = chunk_size - overlap
            for start in range(0, len(segment_indices) - chunk_size + 1, stride):
                emg = segment_emg[:, start:start + chunk_size]  # 取窗口内的信号，确保形状是 (通道数, 采样点)
                clip_id = f'{idx}_{file_name}'

                info = {
                    'clip_id': clip_id,
                    'subject_id': subject_id,
                    'session_id': f'{subject_id}_{session_id}',
                    'trial_id': f'{subject_id}_{session_id}_{hand_gesture}',
                    'stimulus_id': f'{subject_id}_{session_id}_{hand_gesture}_{(seg_id-1) % 6}',
                    'label': hand_gesture
                }
                result = {
                    'key': clip_id,
                    'emg': {
                        'signals': emg,  # 确保最终存储的信号形状是 (通道数, 采样点)
                        'sampling_rate': sampling_rate,
                        'channels': channels
                    }
                }
                result.update({
                    'info': info
                })
                yield result
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
