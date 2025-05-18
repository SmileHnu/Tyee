#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : car.py
@Time    : 2025/03/30 20:35:06
@Desc    : 
"""


from typing import Dict, Any, Optional
import numpy as np
from .base_transform import BaseTransform

import numpy as np
from dataset.transform import BaseTransform

class CommonAverageRef(BaseTransform):
    """
    对每个时间点做 common average referencing（减去所有通道的参考值）。
    支持 median、mean 等多种参考方式。
    """
    def __init__(self, axis=0, mode='median', source=None, target=None):
        """
        axis: 对哪个轴做CAR，通常是通道轴（如0）
        mode: 'median'（中位数）、'mean'（均值），可扩展
        """
        super().__init__(source, target)
        self.axis = axis
        self.mode = mode

    def transform(self, result):
        data = result['data']
        if self.mode == 'median':
            common = np.median(data, axis=self.axis, keepdims=True)
        elif self.mode == 'mean':
            common = np.mean(data, axis=self.axis, keepdims=True)
        else:
            raise ValueError(f"Unknown mode '{self.mode}' for CommonAverageRef. 支持 'median' 或 'mean'")
        data_car = data - common
        result = result.copy()
        result['data'] = data_car
        return result