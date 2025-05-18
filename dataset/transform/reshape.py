#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : reshape.py
@Time    : 2025/05/07 16:49:56
@Desc    : 
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataset.transform import BaseTransform

class Reshape(BaseTransform):
    """
    通用 reshape transform，将数据 reshape 为指定形状。
    """
    def __init__(self, shape: Tuple[int, ...], source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            shape: 目标形状
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.shape = shape

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result = result.copy()
        result['data'] = np.reshape(data, self.shape)
        return result

class Transpose(BaseTransform):
    """
    转置 transform，将数据转置为指定形状。
    """
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            source/target: 指定信号字段
        """
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result = result.copy()
        result['data'] = np.transpose(data)
        return result

class Squeeze(BaseTransform):
    """
    去除数据中指定维度的单维度条目。
    """
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            axis: 指定去除的维度
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.axis = axis
    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result = result.copy()
        result['data'] = np.squeeze(data, axis=self.axis)
        return result

class ExpandDims(BaseTransform):
    """
    扩展数据中指定维度的单维度条目。
    """
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            axis: 指定扩展的维度
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.axis = axis

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result = result.copy()
        result['data'] = np.expand_dims(data, axis=self.axis)
        return result