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
from dataset.transform import BaseTransform

class SlideWindow(BaseTransform):
    def __init__(
        self,
        window_size: int,
        stride: int,
        axis: int = -1,
        source: Optional[str] = None,
        target: Optional[str] = None,
        keep_tail: bool = False
    ):
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
        return result