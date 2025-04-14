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

import math
from typing import Optional, List
from torch.optim import Optimizer
from .base_lr_scheduler import BaseLRScheduler


import math
from typing import Optional, List
from torch.optim import Optimizer
from .base_lr_scheduler import BaseLRScheduler


class CosineLRScheduler(BaseLRScheduler):
    """
    Cosine learning rate scheduler with warmup, restarts, and period multiplication (t_mult).

    This scheduler supports:
    - A linear warmup phase where the learning rate increases from `warmup_start_lr` to `base_lr`
    - A cosine decay phase where the learning rate decreases from `base_lr` to `min_lr`
    - Periodic restarts, where each cycle resets the cosine decay
    - Optional shrinking of the base learning rate after each restart via `lr_shrink`
    - Optional growth of the period length after each restart via `t_mult`

    The learning rate is computed as follows:

    During warmup::
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * step / warmup_steps

    After warmup:
        Within each cosine cycle:
            t = step - last_restart_step
            T = current_period_length
            lr = min_le + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T))

        At each restart:
            base_lr = base_lr * lr_shrink
            current_period_length = current_period_length * t_mult
    """

    def __init__(
        self,
        optimizer: Optimizer,
        niter_per_epoch: int,
        period_steps: Optional[int] = None,
        period_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        min_lr: float = 0.0,
        lr_shrink: float = 1.0,
        t_mult: float = 1.0,
        last_step: int = -1,
    ):
        """
        Initialize the CosineLRScheduler with optional warmup phase and restarts.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            niter_per_epoch (int): Number of iterations per epoch.
            period_steps (int, optional): Total steps for one period of cosine annealing. Default: None.
            period_epochs (int, optional): Total epochs for one period of cosine annealing. Default: None.
            warmup_steps (int, optional): Number of warmup steps. Default: None.
            warmup_epochs (int, optional): Number of warmup epochs. Default: None.
            warmup_start_lr (float): Initial learning rate during warmup. Default: 0.0.
            min_lr (float): Minimum learning rate after cosine annealing. Default: 0.0.
            lr_shrink (float): Shrink factor for learning rate at each restart. Default: 1.0.
            t_mult (float): Factor to grow the length of each period. Default: 1.0.
            last_step (int): The index of the last step. Default: -1.
        """
        if period_steps is None and period_epochs is not None:
            period_steps = period_epochs * niter_per_epoch
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch

        self.niter_per_epoch = niter_per_epoch
        self.period_steps = period_steps
        self.warmup_steps = warmup_steps or 0
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.lr_shrink = lr_shrink
        self.t_mult = t_mult

        self.current_period = period_steps
        self.next_restart_step = self.warmup_steps + self.current_period
        self.last_restart_step = self.warmup_steps

        self.cycle_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        step = self.last_step
        lrs = []

        for i, base_lr in enumerate(self.base_lrs):
            if step < self.warmup_steps:
                # Warmup阶段：学习率线性增加
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
            else:
                # 重启时，更新周期长度和base_lr
                while step >= self.next_restart_step:
                    self.last_restart_step = self.next_restart_step
                    self.current_period = int(self.current_period * self.t_mult)  # 根据 t_mult 增长周期长度
                    self.next_restart_step = self.last_restart_step + self.current_period
                    self.cycle_count += 1
                    # 每次重启时调整 base_lr
                    self.base_lrs[i] *= self.lr_shrink
                    base_lr = self.base_lrs[i]

                # Cosine 衰减阶段
                t = step - self.last_restart_step
                T = self.current_period
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t / T))

            lrs.append(lr)

        return lrs
