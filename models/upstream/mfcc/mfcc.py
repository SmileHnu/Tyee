#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : mfcc.py
@Time    : 2024/11/15 15:12:07
@Desc    : 
"""
import torch
from torch import nn


class MFCC(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forwar(self, sample: torch.Tensor) -> torch.Tensor:
        
        pass