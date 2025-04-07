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
    """
    Compute the norm of gradients for a list of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Iterable of model parameters.
        norm_type (float): The type of norm to compute (default: 2.0).

    Returns:
        float: The computed gradient norm. Returns 0.0 if no gradients are available.
    """
    # Filter parameters that have gradients
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0  # Return 0.0 if no gradients are available

    # Compute the total gradient norm
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)  # Compute the norm of a single parameter's gradient
        total_norm += param_norm.item() ** norm_type

    # Return the total norm raised to the power of 1/norm_type
    return total_norm ** (1.0 / norm_type)
