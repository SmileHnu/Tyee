#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : transpose.py
@Time    : 2024/11/14 16:32:33
@Desc    : 
"""
import torch
import numpy as np


class Transpose(object):
    def __init__(
            self,
            dims: list[int],
        ) -> None:
        self._dims = dims
        pass

    def __call__(self, x) -> torch.Tensor | np.ndarray:
        assert isinstance(x, torch.Tensor) or isinstance(x, np.ndarray), f"Input `x` type isn't valid, expect `torch.Tensor` or `np.ndarray`, while provide {type(x)}"
        if isinstance(x, np.ndarray):
            x = np.transpose(x, axes=self._dims)
        elif isinstance(x, torch.Tensor):
            assert len(self._dims) == 2, f"Only support transpose two dim of `torch.Tensor`, while provide {len(self._dims)}"
            x = torch.transpose(x, dim0=self._dims[0], dim1=self._dims[1])
        return x
