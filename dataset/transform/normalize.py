#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : normalize.py
@Time    : 2025/03/20 10:57:47
@Desc    : 
"""

import numpy as np
from typing import Dict, Any
from dataset.transform import BaseTransform

class Normalize(BaseTransform):
    def __init__(self, method: str = 'quantile', 
                 q: float = 0.95, 
                 epsilon: float = 1e-8,
                 mean: float = None,
                 std: float = None,
                 data_max: float = None,
                 data_min: float = None,
                 low: float = -1,
                 high: float = 1):
        """
        初始化归一化变换类。

        参数:
        - method: 归一化方法，'quantile'、'zscore'、'zscore_per_channel' 或 'min_max'。
        - q: 百分位数，用于百分位数归一化。
        - epsilon: 一个小的常数，用于防止除零错误。
        - mean: 用于zscore归一化的均值。如果为None，则通过signals计算。
        - std: 用于zscore归一化的标准差。如果为None，则通过signals计算。
        - data_max: 用于min_max归一化的最大值。
        - data_min: 用于min_max归一化的最小值。
        - low: min_max归一化的最小值。
        - high: min_max归一化的最大值。
        """
        super().__init__()
        self.method = method
        self.q = q
        self.epsilon = epsilon
        self.mean = mean
        self.std = std
        self.data_max = data_max
        self.data_min = data_min
        self.low = low
        self.high = high

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行归一化处理。

        参数:
        - result: 包含信号数据的字典。

        返回:
        - 更新后的信号数据字典。
        """
        signals = result['signals']
        
        if self.method == 'quantile':
            quantile_value = np.quantile(np.abs(signals), q=self.q, method="linear", axis=-1, keepdims=True)
            result['signals'] = signals / (quantile_value + self.epsilon)
        elif self.method == 'zscore':
            mean = self.mean if self.mean is not None else np.mean(signals)
            std = self.std if self.std is not None else np.std(signals)
            # print("Mean:", mean)
            # print("Std:", std)
            result['signals'] = (signals - mean) / (std + self.epsilon)
        elif self.method == 'zscore_per_channel':
            mean = self.mean if self.mean is not None else np.mean(signals, axis=1, keepdims=True)
            std = self.std if self.std is not None else np.std(signals, axis=1, keepdims=True)
            # print("Mean:", mean)
            # print("Std:", std)
            result['signals'] = (signals - mean) / (std + self.epsilon)
        elif self.method == 'min_max':
            processed_signals = []
            for channel_data in signals:
                xmin = channel_data.min()
                xmax = channel_data.max()
                if xmax - xmin == 0:  # 如果通道数据是常量
                    processed_signals.append(np.zeros_like(channel_data))
                else:
                    # 逐通道归一化
                    normalized = (channel_data - xmin) / (xmax - xmin)
                    normalized -= 0.5
                    normalized += (self.high + self.low) / 2
                    normalized *= (self.high - self.low)
                    processed_signals.append(normalized)
            # 将处理后的通道数据重新组合为 (通道, 采样点) 的形状
            signals = np.array(processed_signals)
            result['signals'] = signals
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        
        return result