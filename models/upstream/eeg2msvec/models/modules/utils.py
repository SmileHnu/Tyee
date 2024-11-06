#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : utils.py
@Time    : 2024/05/10 15:08:17
@Desc    : 
"""
import torch
from torch import nn
from functools import wraps


class Transpose(nn.Module):
    def __init__(self, dim0: int = 0, dim1: int = 1) -> None:
        super().__init__()
        self.dim1 = dim0
        self.dim2 = dim1

    def forward(self, x: torch.Tensor):
        return x.transpose(self.dim1, self.dim2)


class SamePad(nn.Module):
    def __init__(self, kernel_size: int, causal=False) -> None:
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x
    

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
