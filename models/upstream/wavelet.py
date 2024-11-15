#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : wavelet.py
@Time    : 2024/11/15 15:22:33
@Desc    : 
"""
import pywt
import torch
from torch import nn
import numpy as np


class Wavelet(nn.Module):
    def __init__(self, wavelet = 'haar', level = 5):
        super(Wavelet, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        coeffs = []

        for i in range(B):
            batch_coeffs = []
            for j in range(C):
                coeff = pywt.wavedec(x[i, j].cpu().numpy(), self.wavelet, level=self.level)
                # 将小波系数展平，保持在固定长度
                coeff = np.concatenate([c.flatten() for c in coeff])
                
                batch_coeffs.append(coeff)
            # 将当前批次的所有系数堆叠
            coeffs.append(np.stack(batch_coeffs))

        # 将所有的系数合并为一个torch tensor
        coeffs_tensor = torch.tensor(np.stack(coeffs), device=x.device)

        # return shape [B, C, dim]
        return coeffs_tensor.view(B, -1)
    

if __name__ == "__main__":
    wavelet = Wavelet()
    x = torch.randn(32, 1, 750)
    print(wavelet(x).shape)