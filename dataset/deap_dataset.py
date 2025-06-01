#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : deap_dataset.py
@Time    : 2025/05/23 18:59:44
@Desc    : 
"""

import os
import torch
import scipy
import pickle
import numpy as np
from pathlib import Path
from typing import Callable, Union, Dict, List, Tuple, Any
from dataset.base_dataset import BaseDataset

class DEAPDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './DEAP/data_preprocessed_python',
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
    
    def set_records(self, root_path, **kwargs):        
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('.dat'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        return file_list
    
    def read_record(
        self,
        record: tuple | str,
        **kwargs
    ) -> Dict:
        # 1. 数据被降采样到128Hz
        # 2. EOG（眼电图）伪迹被移除，方法如[1]所述
        # 3. 应用了4.0-45.0Hz的带通频率滤波器
        # 4. 数据被平均到了共同的参考电极
        # 5. EEG通道被重新排序，以使它们都按照上面提到的Geneva顺序排列
        # 6. 数据被分割成60秒的试验，并去除了3秒的预试验基线
        # 7. 试验被重新排序，从演示顺序改为视频（Experiment_id）顺序

        with open(record, 'rb') as f:
            x = pickle.load(f, encoding='latin1')
        data = x['data']  # 40 x 40 x 8064 : video/trial x channel x data
        labels = x['labels']  # 40 x 4 : video/trial x label (valence, arousal, dominance, liking)
        freq = 128.0
        eeg_data = data[:, 0:32, :]  # EEG data
        eeg_channels = ['FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
                        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'OZ', 'PZ',
                        'FP2', 'AF4', 'FZ', 'F4', 'F8', 'FC6', 'FC2', 'CZ',
                        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
        eog_data = data[:, 32:34, :]  # EOG data
        eog_channels = ['hEOG', 'vEOG']
        emg_data = data[:, 34:36, :] # EMG data
        emg_channels = ['zEMG', 'tEMG']
        gsr_data = data[:, 36:37, :] # GSR data
        resp_data = data[:, 37:38, :]
        ppg_data = data[:, 38:39, :] # PPG data
        temp_data = data[:, 39:40, :]
        segments = []
        for i in range(len(data)):
            segment = {
                'start': i,
                'end': i,
                'value': {
                    'valence': {
                        'data': labels[i, 0],
                    },
                    'arousal': {
                        'data': labels[i, 1],
                    },
                    'dominance': {
                        'data': labels[i, 2],
                    },
                    'liking': {
                        'data': labels[i, 3],
                    },
                }
            }
            segments.append(segment)
        return {
            'signals': {
                'eeg': {'data': eeg_data, 'freq': freq, 'channels': eeg_channels},
                'eog': {'data': eog_data, 'freq': freq, 'channels': eog_channels},
                'emg': {'data': emg_data,'freq': freq, 'channels': emg_channels},
                'gsr': {'data': gsr_data, 'freq': freq},
                'resp': {'data': resp_data, 'freq': freq},
                'ppg': {'data': ppg_data, 'freq': freq},
                'temp': {'data': temp_data, 'freq': freq}
            },
            'labels': {
                'segments': segments
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(record))[0]
            }
        }

    def process_record(
        self,
        signals,
        labels,
        meta,
        **kwargs
    ):
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
    
    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        对 signals 中所有信号类型按 label['segments'] 分段，返回每段的信号和标签。
        label['segments'] 的 start/end 单位为秒。
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
        # Extract the subject ID from the file name
        # Assuming the file name format is like "subject_id_record_id"
        # You can modify this logic based on your actual file naming convention
        return int(file_name[1:])
    
    def get_segment_id(self, file_name, idx) -> str:
        # Extract the segment ID from the file name
        # Assuming the segment ID is the same as the file name in this case
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        # Extract the trial ID from the index
        # Assuming the trial ID is the same as the index in this case
        return str(idx)

# from dataset.deap_dataset import DEAPDataset
# from dataset.transform import MinMaxNormalize, SlideWindow,Compose, Mapping, OneHotEncode, Concat,Select,Round, ToNumpyInt32
# offline_signal_transform = [
#     Concat(axis=0, source=['gsr', 'resp', 'ppg', 'temp'], target='mulit4'),
#     Compose([
#         MinMaxNormalize(axis=-1),
#         SlideWindow(window_size=128*5, stride=128*3)],
#         source='mulit4', target='mulit4'),
#     Select(key=['mulit4']),
# ]
# offline_label_transform = [
#     Compose([
#         Round(),
#         ToNumpyInt32(),
#         Mapping(mapping={
#             1: 0,
#             2: 1,
#             3: 2,
#             4: 3,
#             5: 4,
#             6: 5,
#             7: 6,
#             8: 7,
#             9: 8,}),
#         # OneHotEncode(num=9),
#     ], source='arousal', target='arousal'),
#     Select(key=['arousal']),
# ]
# import numpy as np
# import pandas as pd
# dataset = DEAPDataset(
#     root_path='/mnt/ssd/lingyus/test',
#     io_path='/mnt/ssd/lingyus/tyee_deap/train',
#     io_chunks=640,
#     offline_label_transform=offline_label_transform,
#     offline_signal_transform=offline_signal_transform,
#     io_mode='hdf5',
#     num_worker=4,
# )