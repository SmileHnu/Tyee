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
from typing import Dict, Any, Optional, Union
from dataset.transform import BaseTransform

class ZScoreNormalize(BaseTransform):
    """
    Z-score normalization supporting a specified axis, with optional custom mean/std.
    """
    def __init__(
        self, 
        mean: Optional[np.ndarray] = None, 
        std: Optional[np.ndarray] = None, 
        axis: Optional[int] = None, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.mean = mean
        self.std = std
        self.axis = axis
        self.epsilon = epsilon

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if (self.mean is None) or (self.std is None):
            mean = np.mean(data, axis=self.axis, keepdims=True) if self.axis is not None else np.mean(data)
            std = np.std(data, axis=self.axis, keepdims=True) if self.axis is not None else np.std(data)
        else:
            mean = self.mean
            std = self.std
            # 自动reshape以适配广播
            if self.axis is not None and isinstance(mean, np.ndarray) and mean.ndim == 1:
                shape = [1] * data.ndim
                shape[self.axis] = -1
                mean = mean.reshape(*shape)
                std = std.reshape(*shape)
        std = np.where(std == 0, 1, std)
        result['data'] = (data - mean) / (std + self.epsilon)
        return result

class MinMaxNormalize(BaseTransform):
    """
    Min-max normalization supporting a specified axis, with optional custom min/max.
    """
    def __init__(
        self, 
        min: Optional[Union[np.ndarray, float]] = None, 
        max: Optional[Union[np.ndarray, float]] = None, 
        axis: Optional[int] = None, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.min = min
        self.max = max
        self.axis = axis

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if (self.min is None) or (self.max is None):
            min_ = np.min(data, axis=self.axis, keepdims=True) if self.axis is not None else np.min(data)
            max_ = np.max(data, axis=self.axis, keepdims=True) if self.axis is not None else np.max(data)
        else:
            min_ = self.min
            max_ = self.max
            if self.axis is not None and isinstance(min_, np.ndarray) and min_.ndim == 1:
                shape = [1] * data.ndim
                shape[self.axis] = -1
                min_ = min_.reshape(*shape)
                max_ = max_.reshape(*shape)
        result['data'] = (data - min_) / (max_ - min_ + 1e-8)
        return result

class QuantileNormalize(BaseTransform):
    """
    Quantile normalization supporting a specified axis.
    """
    def __init__(
        self, 
        q: float = 0.95, 
        axis: Optional[int] = -1, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.q = q
        self.axis = axis
        self.epsilon = epsilon

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        quantile_value = np.quantile(np.abs(data), q=self.q, axis=self.axis, keepdims=True)
        result['data'] = data / (quantile_value + self.epsilon)
        return result