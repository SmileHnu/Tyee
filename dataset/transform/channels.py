#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : channels.py
@Time    : 2025/03/03 15:31:20
@Desc    : 
"""

import numpy as np
from dataset.transform import BaseTransform
from typing import List, Dict, Union

class PickChannels(BaseTransform):
    def __init__(self, channels: List[str]):
        super(PickChannels, self).__init__()
        self.channels = channels

    def transform(self, signal_type: str, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # print('执行了PickChannels')
        channels_key = f'{signal_type}_channels'
        dataset_channels_dict = dict(zip(result[channels_key], list(range(len(result[channels_key])))))
        
        # 检查是否所有目标通道都在当前通道列表中
        missing_channels = [channel for channel in self.channels if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"以下通道在结果字典中未找到: {missing_channels}")
        
        pick_indices = [dataset_channels_dict[channel] for channel in self.channels if channel in dataset_channels_dict]
        
        result[signal_type] = result[signal_type][pick_indices, :]
        result[channels_key] = [result[channels_key][i] for i in pick_indices]
        return result

class OrderChannels(BaseTransform):
    def __init__(self, order: List[str], padding_value: float = 0):
        super(OrderChannels, self).__init__()
        self.channels_order = order
        self.padding_value = padding_value

    def transform(self, signal_type: str, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # print('执行了OrderChannels')
        channels_key = f'{signal_type}_channels'
        dataset_channels_dict = dict(zip(result[channels_key], list(range(len(result[channels_key])))))
        
        # 检查是否所有目标通道都在当前通道列表中
        missing_channels = [channel for channel in self.channels_order if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"以下通道在结果字典中未找到: {missing_channels}")
        
        num_time_points = result[signal_type].shape[1]
        ordered_signal = np.full((len(self.channels_order), num_time_points), self.padding_value, dtype=result[signal_type].dtype)

        for i, channel in enumerate(self.channels_order):
            if channel in dataset_channels_dict:
                ordered_signal[i] = result[signal_type][dataset_channels_dict[channel]]

        result[signal_type] = ordered_signal
        result[channels_key] = self.channels_order
        return result

class ToIndexChannels(BaseTransform):
    def __init__(self, channels: List[str], strict_mode: bool = False):
        super(ToIndexChannels, self).__init__()
        self.channels = channels
        self.strict_mode = strict_mode

    def transform(self, signal_type: str, result: Dict[str, Union[np.ndarray, List[str]]]) -> Dict[str, Union[np.ndarray, List[int]]]:
        channels_key = f'{signal_type}_channels'
        
        dataset_channels_dict = dict(zip(result[channels_key], list(range(len(result[channels_key])))))
        
        if self.strict_mode:
            indices = [self.channels.index(channel) for channel in result[channels_key]]
        else:
            indices = [self.channels.index(channel) for channel in result[channels_key] if channel in self.channels]
        
        result[channels_key] = indices
        
        return result
