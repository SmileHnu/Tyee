#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : pad.py
@Time    : 2025/05/16 10:46:27
@Desc    : 
"""

import numpy as np
from typing import Optional, Any, Dict
from dataset.transform import BaseTransform

class Pad(BaseTransform):
    def __init__(
        self,
        pad_len: int,
        axis: int = 0,
        side: str = 'post',  # 'pre', 'post', 'both'
        mode: str = 'constant',
        constant_values: float = 0,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.pad_len = pad_len
        self.axis = axis
        self.side = side
        self.mode = mode
        self.constant_values = constant_values

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        data = result['data']
        pad_width = [(0, 0)] * data.ndim
        if self.side == 'pre':
            pad_width[self.axis] = (self.pad_len, 0)
        elif self.side == 'post':
            pad_width[self.axis] = (0, self.pad_len)
        elif self.side == 'both':
            pad_width[self.axis] = (self.pad_len, self.pad_len)
        else:
            raise ValueError("side must be 'pre', 'post' or 'both'")
        data_padded = np.pad(
            data,
            pad_width=pad_width,
            mode=self.mode,
            constant_values=self.constant_values if self.mode == 'constant' else None
        )
        result = result.copy()
        result['data'] = data_padded
        return result
