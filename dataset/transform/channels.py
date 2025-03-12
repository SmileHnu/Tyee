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

    def transform(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # print('执行了PickChannels')
        
        dataset_channels_dict = dict(zip(result['channels'], list(range(len(result['channels'])))))
        
        # 检查是否所有目标通道都在当前通道列表中
        missing_channels = [channel for channel in self.channels if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"以下通道在结果字典中未找到: {missing_channels}")
        
        pick_indices = [dataset_channels_dict[channel] for channel in self.channels if channel in dataset_channels_dict]
        
        result['signals'] = result['signals'][pick_indices, :]
        result['channels'] = [result['channels'][i] for i in pick_indices]
        return result

class OrderChannels(BaseTransform):
    def __init__(self, order: List[str], padding_value: float = 0):
        super(OrderChannels, self).__init__()
        self.channels_order = order
        self.padding_value = padding_value

    def transform(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # print('执行了OrderChannels')
        
        dataset_channels_dict = dict(zip(result['channels'], list(range(len(result['channels'])))))
        
        # 检查是否所有目标通道都在当前通道列表中
        missing_channels = [channel for channel in self.channels_order if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"以下通道在结果字典中未找到: {missing_channels}")
        
        num_time_points = result['signals'].shape[1]
        ordered_signal = np.full((len(self.channels_order), num_time_points), self.padding_value, dtype=result['signals'].dtype)

        for i, channel in enumerate(self.channels_order):
            if channel in dataset_channels_dict:
                ordered_signal[i] = result['signals'][dataset_channels_dict[channel]]

        result['signals'] = ordered_signal
        result['channels'] = self.channels_order
        return result

class ToIndexChannels(BaseTransform):
    def __init__(self, channels: List[str], strict_mode: bool = False):
        super(ToIndexChannels, self).__init__()
        self.channels = channels
        self.strict_mode = strict_mode

    def transform(self, result: Dict[str, Union[np.ndarray, List[str]]]) -> Dict[str, Union[np.ndarray, List[int]]]:
        
       
        
        if self.strict_mode:
            indices = [self.channels.index(channel) for channel in result['channels']]
        else:
            indices = [self.channels.index(channel) for channel in result['channels'] if channel in self.channels]
        
        result['channels'] = indices
        
        return result
