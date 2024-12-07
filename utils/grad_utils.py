#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : grad_utils.py
@Time    : 2024/12/07 14:55:09
@Desc    : 
"""

import torch

def get_grad_norm(parameters, norm_type=2.0):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    # 计算所有参数的梯度范数
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)  # 计算单个参数的范数
        total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)
