#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : compose.py
@Time    : 2024/11/14 16:42:33
@Desc    : 
"""
import torch
import numpy as np


class Compose(object):
    def __init__(
            self,
            transforms: list,
        ) -> None:
        self._transforms = transforms

    def __call__(self, sample: dict | torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        for transform in self._transforms:
            sample = transform(sample)
        return sample
