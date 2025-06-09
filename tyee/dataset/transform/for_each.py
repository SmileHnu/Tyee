#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : for_each.py
@Time    : 2025/05/15 21:09:40
@Desc    : 
"""

from dataset.transform import BaseTransform
import numpy as np

class ForEach(BaseTransform):
    """
    Apply base_transform to each segment along the first dimension.
    Suitable for 2D or 3D data (e.g., shape=(N, ...), where N is the number of segments or channels).
    """
    def __init__(self, transforms: BaseTransform, source=None, target=None):
        super().__init__(source, target)
        self.transforms = transforms
        for t in transforms:
            if t.source is not None or t.target is not None:
                raise ValueError("Sub-transforms should not set source or target.")


    def transform(self, result):
        data = result['data']
        segments = []
        for i in range(data.shape[0]):
            seg_result = result.copy()
            seg_result['data'] = data[i]
            for transform in self.transforms:
                seg_result = transform(seg_result)
                # if i==0:
                #     print(transform.__class__.__name__, seg_result['data'])
            segments.append(seg_result['data'])
        result = result.copy()
        result['data'] = np.stack(segments, axis=0)
        return result