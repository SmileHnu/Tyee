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
                min_ = np.expand_dims(min_, axis=self.axis)
                max_ = np.expand_dims(max_, axis=self.axis)
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

class RobustNormalize(BaseTransform):
    """
    Robust normalization: (x - median) / IQR, supports specified axis.
    IQR = q_max percentile - q_min percentile, resistant to outliers.
    Supports unit_variance option to make normalized variance equal to 1.
    """
    def __init__(
        self,
        median: Optional[np.ndarray] = None,
        iqr: Optional[np.ndarray] = None,
        quantile_range: tuple = (25.0, 75.0),
        axis: Optional[int] = None,
        epsilon: float = 1e-8,
        unit_variance: bool = False,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.median = median
        self.iqr = iqr
        self.quantile_range = quantile_range
        self.axis = axis
        self.epsilon = epsilon
        self.unit_variance = unit_variance

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if self.median is None:
            median = np.median(data, axis=self.axis, keepdims=True) if self.axis is not None else np.median(data)
        else:
            median = self.median
            if self.axis is not None and isinstance(median, np.ndarray) and median.ndim == 1:
                median = np.expand_dims(median, axis=self.axis)
        if self.iqr is not None:
            iqr = self.iqr
            if self.axis is not None and isinstance(iqr, np.ndarray) and iqr.ndim == 1:
                iqr = np.expand_dims(iqr, axis=self.axis)
        else:
            q_min, q_max = self.quantile_range
            q_high = np.percentile(data, q_max, axis=self.axis, keepdims=True) if self.axis is not None else np.percentile(data, q_max)
            q_low = np.percentile(data, q_min, axis=self.axis, keepdims=True) if self.axis is not None else np.percentile(data, q_min)
            iqr = q_high - q_low
        iqr = np.where(iqr == 0, 1, iqr)
        # unit_variance缩放
        if self.unit_variance:
            from scipy.stats import norm
            adjust = norm.ppf(q_max / 100.0) - norm.ppf(q_min / 100.0)
            iqr = iqr / adjust
        result = result.copy()
        # print(f'median:{median.shape}, iqr:{iqr.shape}')
        result['data'] = (data - median) / (iqr + self.epsilon)
        return result
    

class Baseline(BaseTransform):
    """
    Apply baseline correction to signals: subtract the mean of a specified interval.
    Supports custom baseline intervals (e.g., first N samples or any interval).
    axis:
        - -1 (default): independent baseline correction for each channel
        - None: use the same baseline for all channels and samples
    """
    def __init__(
        self,
        baseline_start: Optional[int] = None,  # None表示从0开始
        baseline_end: Optional[int] = None,    # None表示到最后
        axis: Optional[int] = -1,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
        """
        baseline_start: starting sample index, None means 0
        baseline_end: ending sample index (exclusive), None means signal end
        axis: which axis to compute mean along, -1 for each channel, None for global
        """
        super().__init__(source, target)
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self.axis = axis

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        # None等价于python切片的默认行为
        baseline_data = data[..., self.baseline_start:self.baseline_end]
        if self.axis is None:
            baseline = baseline_data.mean()
        else:
            baseline = baseline_data.mean(axis=self.axis, keepdims=True)
        result = result.copy()
        result['data'] = data - baseline
        return result

class Mean(BaseTransform):
    def __init__(self, axis: Optional[Union[int, tuple]] = None, source: Optional[str] = None, target: Optional[str] = None, keepdims: bool = False):
        super().__init__(source=source, target=target)
        self.axis = axis
        self.keepdims = keepdims

    def transform(self, result: Union[Dict[str, Any], np.ndarray]) -> np.ndarray:
        data = result['data']
        if self.axis is None:
            data = np.mean(data)
        else:
            data = np.mean(data, axis=self.axis, keepdims=self.keepdims)
            # print(f'Mean: {data.shape}')
        result = result.copy()
        result['data'] = data
        return result