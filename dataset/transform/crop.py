#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : crop.py
@Time    : 2025/05/07 16:27:07
@Desc    : 
"""

from .base_transform import BaseTransform

class Crop(BaseTransform):
    """
    通用裁剪 transform，可指定裁剪起止位置或左右裁剪点数，适用于任意信号。
    """
    def __init__(self, crop_left=0, crop_right=0, axis=-1, source=None, target=None):
        """
        crop_left: 裁剪左侧（起始）多少个点
        crop_right: 裁剪右侧（末尾）多少个点
        axis: 裁剪的轴（通常为时间轴）
        source/target: 指定只对某个信号字段生效
        """
        super().__init__(source, target)
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.axis = axis

    def transform(self, result):
        data = result['data']
        slc = [slice(None)] * data.ndim
        start = self.crop_left
        end = data.shape[self.axis] - self.crop_right if self.crop_right > 0 else None
        slc[self.axis] = slice(start, end)
        data_cropped = data[tuple(slc)]
        result = result.copy()
        result['data'] = data_cropped
        return result