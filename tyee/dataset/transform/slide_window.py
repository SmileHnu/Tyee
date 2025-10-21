#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : slide_window.py
@Time    : 2025/04/19 15:57:26
@Desc    : 
"""

import numpy as np
from typing import Dict, Any, Optional
from tyee.dataset.transform import BaseTransform

class SlideWindow(BaseTransform):
    """
    Sliding window transform that generates window indices and stores them in result['info']['windows'].
    Output format: list of dictionaries with 'start' and 'end' keys.
    """
    def __init__(
        self,
        window_size: int,
        stride: int,
        axis: int = -1,
        source: Optional[str] = None,
        target: Optional[str] = None,
        keep_tail: bool = False
    ):
        """
        Args:
            window_size: size of each window
            stride: step size between windows
            axis: axis along which to apply sliding window
            keep_tail: whether to keep the tail window if it doesn't align with stride
        """
        super().__init__(source, target)
        self.window_size = window_size
        self.stride = stride
        self.axis = axis
        self.keep_tail = keep_tail

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        length = data.shape[self.axis]
        indices = []
        for start in range(0, length - self.window_size + 1, self.stride):
            end = start + self.window_size
            indices.append({'start': start, 'end': end})
        if self.keep_tail and (length - self.window_size) % self.stride != 0:
            start = length - self.window_size
            end = length
            indices.append({'start': start, 'end': end})
        if 'info' not in result or result['info'] is None:
            result['info'] = {}
        result['info']['windows'] = indices
        result['axis'] = self.axis
        return result

class WindowExtract(BaseTransform):
    """
    Extract data slices based on start/end from result['info']['windows'],
    Output shape: (num_windows, channels, window_size)
    """
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        axis = result.get('axis', -1)
        windows_info = result['info']['windows']
        windows = []
        for win in windows_info:
            start, end = win['start'], win['end']
            if axis == -1 or axis == data.ndim - 1:
                windows.append(data[..., start:end])
            else:
                slicer = [slice(None)] * data.ndim
                slicer[axis] = slice(start, end)
                windows.append(data[tuple(slicer)])
        result = result.copy()
        result['data'] = np.stack(windows, axis=0)
        return result