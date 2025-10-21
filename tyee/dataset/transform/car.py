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
from tyee.dataset.transform import BaseTransform

class CommonAverageRef(BaseTransform):
    """
    Apply common average referencing (CAR) to each time point by subtracting the reference value across all channels.
    Supports multiple reference methods such as median, mean, etc.
    """
    def __init__(self, axis=0, mode='median', source=None, target=None):
        """
        axis: Which axis to apply CAR on, typically the channel axis (e.g., 0)
        mode: 'median' (median value), 'mean' (mean value), extensible for other methods
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
            raise ValueError(f"Unknown mode '{self.mode}' for CommonAverageRef. support 'median' 或 'mean'")
        data_car = data - common
        result = result.copy()
        result['data'] = data_car
        return result