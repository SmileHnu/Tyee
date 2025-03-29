#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : convert_signal.py
@Time    : 2025/03/05 14:48:38
@Desc    : 
"""

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : unipolar_to_bipolar_transform.py
@Time    : 2025/03/05 14:14:37
@Desc    : 
"""

import numpy as np
from typing import Dict, Any, List
from dataset.transform import BaseTransform
from utils import lazy_import_module

class UniToBiTransform(BaseTransform):
    def __init__(self, target_channels: List[str]):
        super().__init__()
        if isinstance(target_channels, str):
            self.target_channels = lazy_import_module('dataset.constants', target_channels)
        else:
            self.target_channels = target_channels

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将单极信号转换为双极信号，通过对目标双极通道列表中的相邻通道做差值计算。
        
        参数:
        - result: 包含信号数据的字典。
        - target_bipolar_channels: 目标双极通道列表，格式为 ["channel1-channel2", ...]。
        
        返回:
        - 更新后的信号数据字典。
        """
        signal_data = result['signals']
        channel_list = result['channels']
        
        new_signal_data = []
        new_channel_list = []

        for bipolar_channel in self.target_channels:
            channel1, channel2 = bipolar_channel.split('-')
            if channel1 in channel_list and channel2 in channel_list:
                index1 = channel_list.index(channel1)
                index2 = channel_list.index(channel2)
                bipolar_signal = np.array(signal_data[index1]) - np.array(signal_data[index2])
                new_channel_name = bipolar_channel
                new_signal_data.append(bipolar_signal)
                new_channel_list.append(new_channel_name)
            else:
                raise ValueError(f"在信号数据中未找到通道 {channel1} 和/或 {channel2}。")
        
        new_signal_data = np.array(new_signal_data)
        # 更新通道列表和信号数据
        result['channels'] = new_channel_list
        result['signals'] = new_signal_data
        
        return result