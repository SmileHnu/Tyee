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
    """
    离散化回归损失函数
    功能说明：
    1. 将连续心率真值转换为分箱概率分布（高斯分布）
    2. 计算预测概率分布与目标分布之间的交叉熵损失
    参数说明：
    - dim: 分箱数量
    - min_hz: 最小可预测频率 (Hz)
    - max_hz: 最大可预测频率 (Hz)
    - sigma_y: 目标高斯分布的标准差（控制标签模糊程度）
    """
    def __init__(self, dim: int, min_hz: float, max_hz: float, sigma_y: float):
        super().__init__()
        self.dim = dim
        self.sigma_y = sigma_y
        
        # 将频率转换为BPM并生成分箱边界
        min_bpm = min_hz * 60.0
        max_bpm = max_hz * 60.0
        # 计算分箱边界
        self.bin_edges = torch.arange(
            start=min_bpm, end=max_bpm, step=(max_bpm - min_bpm) / dim
        )

    def _y_to_bins(self, y: torch.Tensor) -> torch.Tensor:
        """
        将连续心率值转换为分箱概率分布
        :param torch.Tensor y: 真实心率值张量，形状 (batch_size, seq_len)
        :return torch.Tensor: 分箱概率分布，形状 (batch_size, seq_len, dim)
        """
        # 扩展维度以支持广播计算
        y = y.unsqueeze(-1)  # (batch, seq, 1)
        bins = self.bin_edges.unsqueeze(0).unsqueeze(0).to(y.device)  # (1, 1, dim)
        
        # 计算高斯概率密度
        dist = torch.distributions.Normal(loc=y, scale=self.sigma_y)
        probs = torch.exp(dist.log_prob(bins))  # (batch, seq, dim)
        
        # 沿分箱维度归一化
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        :param torch.Tensor y_pred: 模型预测概率分布，形状 (batch_size, seq_len, dim), 需要经过softmax处理
        :param torch.Tensor y_true: 真实心率值，形状 (batch_size, seq_len)
        :return torch.Tensor: 标量损失值
        """
        # 转换真值为分箱概率分布
        binned_true = self._y_to_bins(y_true)  # (batch, seq, dim)
                
        # 计算交叉熵损失
        losses = - (binned_true * torch.log(y_pred + 1e-10)).sum(dim=-1)
        return losses.mean()
    
    def extra_repr(self) -> str:
        """用于打印实例信息"""
        return f"dim={self.dim}, sigma_y={self.sigma_y}, bins=[{self.bin_edges[0]:.1f}-{self.bin_edges[-1]:.1f}bpm]"
