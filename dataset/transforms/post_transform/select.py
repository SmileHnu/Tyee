#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : select.py
@Time    : 2024/11/14 16:52:44
@Desc    : 
"""
import torch
import inspect
import numpy as np


class Select(object):
    def __init__(
            self,
            channel: str,
            ref_channels: list = None
        ) -> None:
        self._channel = channel
        self._ref_channels = ref_channels
        pass

    def __call__(self, x) -> torch.Tensor:
        idx = self._ref_channels.index(self._channel)
        if idx >=0 and idx < len(self._ref_channels):
            if isinstance(x, torch.Tensor):
                return x[idx, ...].unsqueeze(0)
            elif isinstance(x, np.ndarray):
                return x[idx, ...][np.newaxis, :]
            else:
                return [x[idx, ...]]
        else:
            raise Exception("Input channel name not found in ref channels")

