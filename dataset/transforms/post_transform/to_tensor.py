#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : to_tensor.py
@Time    : 2024/11/14 16:40:00
@Desc    : 
"""
import torch


class ToTensor(object):
    def __init__(
            self,
        ) -> None:
        pass

    def __call__(self, x) -> torch.Tensor:
        return torch.tensor(x)
