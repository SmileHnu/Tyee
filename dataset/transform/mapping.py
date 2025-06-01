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
from .base_transform import BaseTransform

class Mapping(BaseTransform):
    def __init__(self, mapping: dict, source: str = None, target: str = None):
        super().__init__(source, target)
        self.mapping = mapping

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        value = result['data']
        
        # 如果是标量，直接映射
        if np.isscalar(value):
            result['data'] = self.mapping[value]
        else:
            # 如果是数组或可迭代对象，使用numpy的向量化映射
            if isinstance(value, np.ndarray):
                # 保存原始形状
                original_shape = value.shape
                flat_value = value.flatten()
                
                # 向量化映射
                mapped_flat = np.array([self.mapping[item] for item in flat_value])
                
                # 恢复形状
                result['data'] = mapped_flat.reshape(original_shape)
            else:
                # 处理其他可迭代类型
                result['data'] = np.array([self.mapping[item] for item in value])
        
        return result