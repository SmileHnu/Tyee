#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : binned_regression_criterion.py
@Time    : 2025/03/30 20:33:12
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinnedRegressionLoss(nn.Module):
    def __init__(self, dim: int, min_hz: float, max_hz: float, sigma_y: float):
        super().__init__()
        self.dim = dim
        self.sigma_y = sigma_y
        
        min_bpm = min_hz * 60.0
        max_bpm = max_hz * 60.0
 
        self.bin_edges = torch.arange(
            start=min_bpm, end=max_bpm, step=(max_bpm - min_bpm) / dim
        )

    def _y_to_bins(self, y: torch.Tensor) -> torch.Tensor:

        y = y.unsqueeze(-1)  # (batch, seq, 1)
        bins = self.bin_edges.unsqueeze(0).unsqueeze(0).to(y.device)  # (1, 1, dim)
        
        dist = torch.distributions.Normal(loc=y, scale=self.sigma_y)
        log_probs_before_exp = dist.log_prob(bins)
        probs = torch.exp(log_probs_before_exp)

        
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        binned_true = self._y_to_bins(y_true)  # (batch, seq, dim)
                
        log_y_pred_stable = torch.log(y_pred + 1e-10)
        log_y_pred_expanded = log_y_pred_stable.unsqueeze(1)
        product = binned_true * log_y_pred_expanded
        losses = - product.sum(dim=-1)
        return losses.mean()
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, sigma_y={self.sigma_y}, bins=[{self.bin_edges[0]:.1f}-{self.bin_edges[-1]:.1f}bpm]"
