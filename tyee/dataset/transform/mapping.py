#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mapping.py
@Time    : 2025/05/07 10:17:45
@Desc    : 
"""

from typing import Dict, Any
import numpy as np
from tyee.dataset.transform import BaseTransform

class Mapping(BaseTransform):
    def __init__(self, mapping: dict, source: str = None, target: str = None):
        super().__init__(source, target)
        self.mapping = mapping

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        value = result['data']
        
        if np.isscalar(value):
            result['data'] = self.mapping[value]
        else:
            if isinstance(value, np.ndarray):
                original_shape = value.shape
                flat_value = value.flatten()
                
                mapped_flat = np.array([self.mapping[item] for item in flat_value])
                
                result['data'] = mapped_flat.reshape(original_shape)
            else:
                result['data'] = np.array([self.mapping[item] for item in value])
        
        return result