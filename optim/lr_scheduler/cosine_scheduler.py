#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : cosine_scheduler.py
@Time    : 2024/12/09 20:20:28
@Desc    : 
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

class CosineScheduler(SequentialLR):
    def __init__(self, optimizer, T_max, warmup_steps, warmup_start_lr=0, eta_min=0, last_epoch=-1):
        """
        初始化 CosineScheduler，使用 LinearLR 和 CosineAnnealingLR 组合实现预热和余弦退火。

        :param optimizer: 优化器
        :param T_max: CosineAnnealing 的最大步数
        :param warmup_steps: 预热阶段的步数
        :param warmup_start_lr: 预热阶段的初始学习率
        :param eta_min: 余弦退火阶段的最小学习率
        :param last_epoch: 上一次更新的步数，默认为-1
        """
        base_lr = optimizer.param_groups[0]['lr']
        warmup_lr_schedule = LinearLR(optimizer, start_factor=warmup_start_lr/base_lr, total_iters=warmup_steps)
        cosine_lr_schedule = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        super(CosineScheduler, self).__init__(optimizer, schedulers=[warmup_lr_schedule, cosine_lr_schedule], milestones=[warmup_steps])