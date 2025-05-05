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
from typing import Dict, Any, List, Union
from dataset.transform import BaseTransform
from utils import lazy_import_module

class UniToBiTransform(BaseTransform):
    def __init__(self, target_channels: Union[str, List[str]], source: str = None, target: str = None):
        super().__init__(source, target)
        if isinstance(target_channels, str):
            self.target_channels = lazy_import_module('dataset.constants', target_channels)
        else:
            self.target_channels = target_channels

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert unipolar signals to bipolar signals by calculating the difference between adjacent channels
        specified in the target_channels list.

        Args:
            result (Dict[str, Any]): Dictionary containing signal data with keys 'data', 'channels', and 'freq'.
                - 'data': 2D array-like, shape (n_channels, n_times)
                - 'channels': List of channel names
                - 'freq': (optional) Sampling frequency

        Returns:
            Dict[str, Any]: Updated signal dictionary with bipolar signals.
                - 'data': 2D array, shape (n_bipolar_channels, n_times)
                - 'channels': List of bipolar channel names (e.g., ["A-B", ...])
                - 'freq': Sampling frequency (if present in input)
        """
        signal_data = result['data']
        channel_list = result['channels']

        new_signal_data = []
        new_channel_list = []

        for bipolar_channel in self.target_channels:
            channel1, channel2 = bipolar_channel.split('-')
            if channel1 in channel_list and channel2 in channel_list:
                index1 = channel_list.index(channel1)
                index2 = channel_list.index(channel2)
                bipolar_signal = np.array(signal_data[index1]) - np.array(signal_data[index2])
                new_signal_data.append(bipolar_signal)
                new_channel_list.append(bipolar_channel)
            else:
                raise ValueError(f"Channel {channel1} and/or {channel2} not found in the signal data.")

        new_signal_data = np.array(new_signal_data)
        # Return the new signal dictionary
        return {
            'data': new_signal_data,
            'channels': new_channel_list,
            'freq': result.get('freq', None)
        }