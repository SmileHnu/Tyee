#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : lambd.py
@Time    : 2025/03/18 19:12:00
@Desc    : 
"""

from typing import Callable
from dataset.transform import BaseTransform

class Lambda(BaseTransform):
    def __init__(self, lambd: Callable):
        super().__init__()
        self.lambd = lambd
    
    def transform(self, result):
        return self.lambd(result)