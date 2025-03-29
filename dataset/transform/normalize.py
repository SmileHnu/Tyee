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
            mean = self.mean if self.mean is not None else np.mean(signals, axis=0, keepdims=True)
            std = self.std if self.std is not None else np.std(signals, axis=0, keepdims=True)
            # print("Mean:", mean)
            # print("Std:", std)
            result['signals'] = (signals - mean) / (std + self.epsilon)
        elif self.method == 'min_max':
            if self.data_max is not None and self.data_min is not None:
                max_scale = self.data_max - self.data_min
                scale = 2 * (np.clip((signals.max() - signals.min()) / max_scale, 0, 1) - 0.5)
            if len(signals.shape) == 2:
                xmin = signals.min()
                xmax = signals.max()
                if xmax - xmin == 0:
                    signals = np.zeros_like(signals)
                    return result
            elif len(signals.shape) == 3:
                xmin = np.min(np.min(signals, axis=1, keepdims=True), axis=-1, keepdims=True)
                xmax = np.max(np.max(signals, axis=1, keepdims=True), axis=-1, keepdims=True)
                constant_trials = (xmax - xmin) == 0
                if np.any(constant_trials):
                    xmax[constant_trials] += 1e-6

            signals = (signals - xmin) / (xmax - xmin)
            signals -= 0.5
            signals += (self.high + self.low) / 2
            signals *= (self.high - self.low)
            if self.data_max is not None:
                signals = np.concatenate([signals, np.ones((1, signals.shape[-1])) * scale], axis=0)
            result['signals'] = signals
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")
        
        return result