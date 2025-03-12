#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : compose.py
@Time    : 2025/03/03 14:30:17
@Desc    : 
"""


from dataset.transform import BaseTransform

class Compose(BaseTransform):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, result):
        for t in self.transforms:
            result = t.transform(result)
        return result