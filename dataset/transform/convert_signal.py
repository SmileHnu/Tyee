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

TCP_MONTAGE_CHANNELS = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "A1-T3", "T3-C3", "C3-CZ", 
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T4-A2", "C4-T4", "CZ-C4",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

class UniToBiTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, result: Dict[str, Any], target_bipolar_channels: List[str]) -> Dict[str, Any]:
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
        
        new_signal_data = {}
        new_channel_list = []

        for bipolar_channel in target_bipolar_channels:
            channel1, channel2 = bipolar_channel.split('-')
            if channel1 in signal_data and channel2 in signal_data:
                bipolar_signal = np.array(signal_data[channel1]) - np.array(signal_data[channel2])
                new_channel_name = f'bipolar_{channel1}_{channel2}'
                new_signal_data[new_channel_name] = bipolar_signal
                new_channel_list.append(new_channel_name)
            else:
                raise ValueError(f"在信号数据中未找到通道 {channel1} 和/或 {channel2}。")
        
        # 更新通道列表和信号数据
        result['channels'] = new_channel_list
        result['signals'] = new_signal_data
        
        return result