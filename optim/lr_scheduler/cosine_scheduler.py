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
import math
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

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_steps, warmup_start_lr=0, eta_min=0, last_epoch=-1):
        """
        初始化 Warmup 和 CosineAnnealing 的学习率调度器，使用 lr_scale 而不是 layer_decay 来调整学习率。

        :param optimizer: 优化器
        :param T_max: CosineAnnealing 的最大步数
        :param warmup_steps: 预热阶段的步数
        :param warmup_start_lr: 预热阶段的初始学习率
        :param eta_min: 余弦退火阶段的最小学习率
        :param last_epoch: 上一次更新的步数，默认为-1
        """
        self.warmup_steps = warmup_steps  # 预热的步数
        self.T_max = T_max  # CosineAnnealing 的总步数
        self.eta_min = eta_min  # 最小学习率
        self.warmup_start_lr = warmup_start_lr  # 预热的起始学习率
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 获取当前步数
        step = self.last_epoch
        base_lr = self.base_lrs[0]  # 假设所有层使用相同的基本学习率

        # 线性 warmup 阶段
        if step < self.warmup_steps:
            lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
        else:
            # CosineAnnealing 阶段
            step_after_warmup = step - self.warmup_steps
            lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * step_after_warmup / (self.T_max - self.warmup_steps)))

        # 获取每层的 lr_scale
        layer_lr_list = []
        for param_group in self.optimizer.param_groups:
            
            # 计算该层的学习率，使用 lr_scale 来调整
            lr_scale = param_group.get('lr_scale', 1.0)  # 默认 lr_scale 为 1.0
            adjusted_lr = lr * lr_scale  # 使用 lr_scale 调整学习率
            # 更新该层的学习率
            layer_lr_list.append(adjusted_lr)

        # 返回每个参数组的最终学习率
        return layer_lr_list