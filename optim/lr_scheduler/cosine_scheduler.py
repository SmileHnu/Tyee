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

class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, 
                 optimizer, 
                 niter_per_epoch,
                 total_steps=None, 
                 total_epochs=None,
                 warmup_steps=0, 
                 warmup_epochs=0,
                 warmup_start_lr=0, 
                 eta_min=0, 
                 last_epoch=-1):
        """
        初始化 Warmup 和 CosineAnnealing 的学习率调度器，使用 lr_scale 而不是 layer_decay 来调整学习率。

        :param optimizer: 优化器
        :param total_steps: CosineAnnealing 的总步数
        :param total_epochs: CosineAnnealing 的总周期数
        :param warmup_steps: 预热阶段的步数
        :param warmup_epochs: 预热阶段的周期数
        :param niter_per_epoch: 每个周期的迭代次数
        :param warmup_start_lr: 预热阶段的初始学习率
        :param eta_min: 余弦退火阶段的最小学习率
        :param last_epoch: 上一次更新的步数，默认为-1
        """
        if total_steps is None and total_epochs is not None:
            total_steps = total_epochs * niter_per_epoch
        if warmup_steps == 0 and warmup_epochs > 0 :
            warmup_steps = warmup_epochs * niter_per_epoch

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

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
            lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * step_after_warmup / (self.total_steps - self.warmup_steps)))

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