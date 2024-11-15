#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : squeeze.py
@Time    : 2024/11/14 16:57:54
@Desc    : 
"""
import torch


class Unsqueeze(object):
    def __init__(
            self,
            dim: int
        ) -> None:
        self._dim = dim
        pass

    def __call__(self, x) -> torch.Tensor:
        return x.unsqueeze(self._dim)


class Squeeze(object):
    def __init__(
            self,
            dim: int
        ) -> None:
        self._dim = dim
        pass

    def __call__(self, x) -> torch.Tensor:
        return x.Ssqueeze(self._dim)