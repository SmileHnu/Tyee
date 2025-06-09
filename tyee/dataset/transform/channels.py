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
from typing import List, Dict, Union, Any
from utils import lazy_import_module

class PickChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], source: str = None, target: str = None):
        """
        Initialize PickChannels transform.

        Args:
            channels (Union[str, List[str]]): List of channel names or a string key to import.
            source (str, optional): Source key for the signal.
            target (str, optional): Target key for the transformed signal.
        """
        super().__init__(source, target)
        if isinstance(channels, str):
            self.channels = lazy_import_module('dataset.constants', channels)
        else:
            self.channels = channels

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pick specified channels from the input signal.

        Args:
            result (Dict[str, Any]): Input dictionary containing 'data' and 'channels'.

        Returns:
            Dict[str, Any]: Dictionary with selected channels and corresponding data.
        """
        dataset_channels_dict = dict(zip(result['channels'], range(len(result['channels']))))
        missing_channels = [channel for channel in self.channels if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"The following channels are not found in the result dictionary: {missing_channels}")
        pick_indices = [dataset_channels_dict[channel] for channel in self.channels]
        result['data'] = result['data'][pick_indices, :]
        result['channels'] = [result['channels'][i] for i in pick_indices]
        return result

class OrderChannels(BaseTransform):
    def __init__(self, order: Union[str, List[str]], padding_value: float = 0, source: str = None, target: str = None):
        """
        Initialize OrderChannels transform.

        Args:
            order (Union[str, List[str]]): Desired channel order or a string key to import.
            padding_value (float, optional): Value to use for missing channels. Default is 0.
            source (str, optional): Source key for the signal.
            target (str, optional): Target key for the transformed signal.
        """
        super().__init__(source, target)
        if isinstance(order, str):
            self.channels_order = lazy_import_module('dataset.constants', order)
        else:
            self.channels_order = order
        self.padding_value = padding_value

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reorder channels according to the specified order, padding missing channels if necessary.

        Args:
            result (Dict[str, Any]): Input dictionary containing 'data' and 'channels'.

        Returns:
            Dict[str, Any]: Dictionary with reordered channels and corresponding data.
        """
        dataset_channels_dict = dict(zip(result['channels'], range(len(result['channels']))))
        missing_channels = [channel for channel in self.channels_order if channel not in dataset_channels_dict]
        if missing_channels:
            raise KeyError(f"The following channels are not found in the result dictionary: {missing_channels}")
        num_time_points = result['data'].shape[1]
        ordered_data = np.full((len(self.channels_order), num_time_points), self.padding_value, dtype=result['data'].dtype)
        for i, channel in enumerate(self.channels_order):
            if channel in dataset_channels_dict:
                ordered_data[i] = result['data'][dataset_channels_dict[channel]]
        result['data'] = ordered_data
        result['channels'] = self.channels_order
        return result

class ToIndexChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], strict_mode: bool = False, source: str = None, target: str = None):
        """
        Initialize ToIndexChannels transform.

        Args:
            channels (Union[str, List[str]]): List of channel names or a string key to import.
            strict_mode (bool, optional): If True, all channels must be present. Default is False.
            source (str, optional): Source key for the signal.
            target (str, optional): Target key for the transformed signal.
        """
        super().__init__(source, target)
        if isinstance(channels, str):
            self.channels = lazy_import_module('dataset.constants', channels)
        else:
            self.channels = channels
        self.strict_mode = strict_mode

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert channel names to their corresponding indices.

        Args:
            result (Dict[str, Any]): Input dictionary containing 'channels'.

        Returns:
            Dict[str, Any]: Dictionary with channels replaced by their indices.
        """
        if self.strict_mode:
            indices = [self.channels.index(channel) for channel in result['channels']]
        else:
            indices = [self.channels.index(channel) for channel in result['channels'] if channel in self.channels]
        result['channels'] = indices
        return result
