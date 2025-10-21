#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : to.py
@Time    : 2025/05/21 14:44:14
@Desc    : 
"""
import torch
import numpy as np
import matplotlib as mpl
from torchvision import transforms
from typing import Dict, Any
from tyee.dataset.transform import BaseTransform

class ToImage(BaseTransform):
    def __init__(self, length: int, width: int, resize_length_factor: float, native_resnet_size: int,
                 cmap: str = 'viridis', source: str = None, target: str = None):
        super().__init__(source, target)
        self.length = length
        self.width = width
        self.resize_length_factor = resize_length_factor
        self.native_resnet_size = native_resnet_size
        self.cmap = cmap = mpl.colormaps[cmap]

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        numElectrodes = len(result['channels'])
        image = self.optimized_makeOneImage(data, self.cmap, self.length, self.width,
                                       self.resize_length_factor, self.native_resnet_size, numElectrodes)
        result['data'] = image
        return result
    
    def optimized_makeOneImage(self, data, cmap, length, width, resize_length_factor, native_resnet_size, numElectrodes):
        # Contrast normalize and convert data
        # NOTE: Should this be contrast normalized? Then only patterns of data will be visible, not absolute values
        data = (data - data.min()) / (data.max() - data.min())
        data_converted = cmap(data)
        rgb_data = data_converted[:, :3]
        image_data = np.reshape(rgb_data, (numElectrodes, width, 3))
        image = np.transpose(image_data, (2, 0, 1))
        
        # Resize image
        resize = transforms.Resize([length * resize_length_factor, native_resnet_size],
                                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        image = resize(torch.from_numpy(image))
        
        # Get max and min values after interpolation
        max_val = image.max()
        min_val = image.min()
        
        # Contrast normalize again after interpolation
        image = (image - min_val) / (max_val - min_val)
        
        # Normalize with standard ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        return image.numpy().astype(np.float32)

class ToTensor(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result['data'] = torch.from_numpy(data)
        return 

class ToTensorFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        result['data'] = data.float()
        return result

class ToTensorFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        result['data'] = data.half()
        return result

class ToTensorInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        result['data'] = data.long()
        return result
    
class ToNumpy(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        result['data'] = data.numpy()
        return result

class ToNumpyFloat64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        result['data'] = data.astype(np.float64)
        return result

class ToNumpyFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        result['data'] = data.astype(np.float32)
        return result

class ToNumpyFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        result['data'] = data.astype(np.float16)
        return result

class ToNumpyInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        result['data'] = data.astype(np.int64)
        return result

class ToNumpyInt32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        result['data'] = data.astype(np.int32)
        return result
