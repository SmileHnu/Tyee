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
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class NinaproDB5Dataset(BaseDataset):
    def __init__(
        self,
        # NinaproDB5 specific parameters
        wLen: float = 0.25,
        stepLen: float = 0.05,
        balance: bool = True,
        include_transitions: bool = False,
        # BaseDataset parameters
        root_path: str = './NinaProDB5',
        start_offset: float = 0,
        end_offset: float = 0,
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
            # BaseDataset parameters
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
        # NinaproDB5 specific parameters
        self.wlen = wLen
        self.steplen = stepLen
        self.balance = balance
        self.include_transitions = include_transitions
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

    def read_record(self, record: str, **kwargs):
        
        a_data = scio.loadmat(record)
        emg_data = np.array(a_data['emg']).astype(np.float64).T
        restimulus = np.array(a_data['restimulus'])
        freq = float(np.squeeze(a_data['frequency']))
        print(freq)

        emg = {
            'data': emg_data,
            'freq': freq,
            'channels': [f'{i}' for i in range(1, emg_data.shape[0] + 1)],
        }
        windows = []
        gestures = []
        for idx in range(0, restimulus.shape[0] - int(self.wlen * freq) + 1, int(self.steplen * freq)):
            start = idx / freq
            end = start + self.wlen
            # print(f"start: {start}, end: {end}")
            windows.append({
                'start': start,
                'end': end,
            })
            element = restimulus[idx:idx + int(self.wlen * freq)]
            gestures.append(element)
        if self.balance:
            indices = self.balance_gesture_classifier(gestures, self.include_transitions)
            # print(f"balance indices: {len(indices)}")
            # print(f"total indices: {len(gestures)}")
            # return
            gestures = [gestures[i] for i in indices]
            windows = [windows[i] for i in indices]
        gestures = self.contract_gesture_classifier(gestures, self.include_transitions)

        assert len(gestures) == len(windows), f"labels and windows length mismatch: {len(gestures)} != {len(windows)}"
        segments = []
        file_name = os.path.splitext(os.path.basename(record))[0]
        exercise = int(file_name.split('_')[1][1:])
        # print(f"exercise: {exercise}")
        # return
        exercise_idx = [0, 12, 12+17]
        for idx, window in enumerate(windows):
            start = window['start']
            end = window['end']
            segment = {
                'start': start,
                'end': end,
                'value': {
                    'gesture':{
                        'data': gestures[idx] if gestures[idx] == 0 else gestures[idx] + exercise_idx[exercise-1],
                    }
                }
            }
            segments.append(segment)
        
        return {
            'signals':{
                'emg': emg,
            },
            'labels':{
                'segments': segments,
            },
            'meta':{
                'file_name': file_name,
            }

        }

    def update_segment_info(self, seg_info: Dict, meta: Dict, idx: int, segment: Dict):
        seg_info['gesture'] = segment['labels']['gesture']['data']

    def get_subject_id(self, file_name) -> str:
        return file_name.split('_')[0][1:]
    
    def get_session_id(self, file_name) -> str:
        return file_name.split('_')[1][1:]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)

    def balance_gesture_classifier(self, restimulus, include_transitions=False):
        """ Balances distribution of restimulus by minimizing zero (rest) gestures.

        Args:
            restimulus (tensor): restimulus tensor
            args: argument parser object

        """
        numZero = 0
        indices = []
        count_dict = {}
        
        # First pass: count the occurrences of each unique tensor
        for x in range(len(restimulus)):
            unique_elements = np.unique(restimulus[x])
            if len(unique_elements) == 1:
                element = unique_elements.item()
                element = (element, )
                if element in count_dict:
                    count_dict[element] += 1
                else:
                    count_dict[element] = 1

            else:
                if include_transitions:
                    elements = (restimulus[x][0][0].item(), restimulus[x][0][-1].item()) # take first and last gesture (transition window)

                    if elements in count_dict:
                        count_dict[elements] += 1
                    else:
                        count_dict[elements] = 1
                    
        # Calculate average count of non-zero elements
        non_zero_counts = [count for key, count in count_dict.items() if key != (0,)]
        print(count_dict)
        print(non_zero_counts)
        print(sum(non_zero_counts))
        if non_zero_counts:
            avg_count = sum(non_zero_counts) / len(non_zero_counts)
        else:
            avg_count = 0  # Handle case where there are no non-zero unique elements
        print(f"avg_count: {avg_count}")
        for x in range(len(restimulus)):
            unique_elements = np.unique(restimulus[x])
            if len(unique_elements) == 1:
                gesture = unique_elements.item()
                if gesture == 0: 
                    if numZero < avg_count:
                        indices.append(x)
                    numZero += 1 # Rest always in partial
                else:
                    indices.append(x)
            else:
                if include_transitions:
                    indices.append(x)
        return indices
    
    def contract_gesture_classifier(self, restim, include_transitions=False):
        labels = []
        for x in range(len(restim)):
            if include_transitions:
                gesture = int(restim[x][-1])  
            else:
                gesture = int(restim[x][0]) 
            labels.append(gesture)

        return labels


# from dataset.ninapro_db5_dataset import NinaproDB5Dataset
# from dataset.transform import Mapping, Filter, ImageResize, NotchFilter, ToImage, ToNumpyFloat16,Reshape, OneHotEncode
# onffline_label_transform = [
#     Mapping(mapping={
#         0: 0,
#         17: 1,
#         18: 2,
#         20: 3,
#         21: 4,
#         22: 5,
#         25: 6,
#         26: 7,
#         27: 8,
#         28: 9,
#     }, source='gesture', target='gesture'),
#     OneHotEncode(num=10, source='gesture', target='gesture'),
# ]
# offline_signal_transform = [
#     Filter(h_freq=None, l_freq=5.0, method='iir', iir_params=dict(order=3, ftype='butter', padlen=12), phase='zero', source='emg', target='emg'),
#     # NotchFilter(freqs=[50.0], method='iir', source='emg', target='emg'),
#     Reshape(shape=16*50, source='emg', target='emg'),
#     ToImage(length=16, width=50, resize_length_factor=1, native_resnet_size=224, cmap='viridis', source='emg', target='emg'),
#     ToNumpyFloat16(source='emg', target='emg'),
# ]
# online_signal_transform = [
#     ImageResize(size=(224,224), source='emg', target='emg')
# ]
# dataset = NinaproDB5Dataset(
#     root_path='/mnt/ssd/lingyus/NinaproDB5E2',
#     io_path='/mnt/ssd/lingyus/tyee_ninapro_db5/train',
#     # io_chunks=224,
#     offline_label_transform=onffline_label_transform,
#     offline_signal_transform=offline_signal_transform,
#     # online_signal_transform=online_signal_transform,
#     io_mode='hdf5',
#     num_worker=4,
# )
