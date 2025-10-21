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
from tyee.dataset.transform import BaseTransform

class Reshape(BaseTransform):
    """
    General reshape transform, reshapes data to specified shape.
    """
    def __init__(self, shape: Tuple[int, ...], source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            shape: target shape
            source/target: specify signal field
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
    Transpose transform, transposes data to specified shape.
    """
    def __init__(self, axes=None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            axes: axes order for transposition
            source/target: specify signal field
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
    Remove single-dimensional entries from the shape of data along specified axis.
    """
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            axis: axis along which to squeeze
            source/target: specify signal field
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
    Expand the shape of data by inserting new axes along specified axis.
    """
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            axis: position in the expanded axes where the new axis is placed
            source/target: specify signal field
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
    Transform that inserts single values at specified positions.
    """
    def __init__(self, 
                 indices: Union[int, List[int], np.ndarray], 
                 value: Union[int, float] = 0,
                 axis: int = 1, 
                 source: Optional[str] = None, 
                 target: Optional[str] = None):
        """
        Args:
            indices: indices where values are inserted, can be single integer or list of indices
            value: single numerical value to insert, default is 0
            axis: axis along which to insert, default is 1
            source/target: specify signal field
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
    Image resize transform, resizes image to specified shape.
    """
    def __init__(self, size: Tuple[int, int], source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            size: target size
            source/target: specify signal field
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