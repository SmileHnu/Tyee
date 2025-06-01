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

from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import torch
from torchvision.transforms import Resize
from PIL import Image
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
    def __init__(self, axes=None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.axes = axes

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result = result.copy()
        if self.axes is not None:
            result['data'] = np.transpose(data, self.axes)
        else:
            result['data'] = np.transpose(data)  # 完全反转
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

class Insert(BaseTransform):
    """
    在指定位置插入单个值的 transform。
    """
    def __init__(self, 
                 indices: Union[int, List[int], np.ndarray], 
                 value: Union[int, float] = 0,
                 axis: int = 1, 
                 source: Optional[str] = None, 
                 target: Optional[str] = None):
        """
        Args:
            indices: 插入位置的索引，可以是单个整数或索引列表
            value: 要插入的单个数值，默认为0
            axis: 插入的维度，默认为1
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.indices = indices if isinstance(indices, (list, np.ndarray)) else [indices]
        self.value = value
        self.axis = axis

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result_copy = result.copy()
        
        for idx in self.indices:
            data = np.insert(data, idx, self.value, axis=self.axis)
        
        result_copy['data'] = data
        return result_copy
    
class ImageResize(BaseTransform):
    """
    图像 resize transform，将图像 resize 为指定形状。
    """
    def __init__(self, size: Tuple[int, int], source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            size: 目标形状
            source/target: 指定信号字段
        """
        super().__init__(source, target)
        self.size = size

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result_copy = result.copy()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        resize = Resize(self.size)
        data = resize(data)

        if isinstance(data, Image.Image):
                data = np.array(data)
        elif isinstance(data, torch.Tensor):
            # Make sure the tensor is in CPU and convert it
            data = np.float32(data.cpu().detach().numpy())
        
        result_copy['data'] = data
        return result_copy