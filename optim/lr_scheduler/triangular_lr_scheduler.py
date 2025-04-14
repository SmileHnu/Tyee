#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : triangular_lr_scheduler.py
@Time    : 2025/04/11 14:03:08
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


class TriangularLRScheduler(BaseLRScheduler):
    """
    Triangular Learning Rate Scheduler with cycle multiplication (t_mult) and shrink.

    Implements a cyclical learning rate schedule where:
    - LR increases linearly from `min_lr` to `max_lr` in the first half of the cycle,
    - and decreases linearly back to `min_lr` in the second half.

    Supports:
    - shrinking of `max_lr` and optionally `min_lr` by `lr_shrink` after each cycle,
    - increasing the length of each cycle by `t_mult`.

    During a cycle:
        lr = min_lr + (max_lr - min_lr) * max(0, 1 - abs(t_curr / period - 1))

    After each cycle:
        max_lr = max_lr * lr_shrink
        min_lr = min_lr * lr_shrink (if `shrink_min`)
        period = period * t_mult
    """

    def __init__(
        self,
        optimizer: Optimizer,
        niter_per_epoch: int,
        period_epochs: Optional[int] = None,
        period_steps: Optional[int] = None,
        min_lr: float = 0.0,
        lr_shrink: float = 1.0,
        shrink_min: bool = False,
        t_mult: float = 1.0,
        last_step: int = -1,
    ):
        if period_steps is None and period_epochs is not None:
            period_steps = period_epochs * niter_per_epoch

        self.niter_per_epoch = niter_per_epoch
        self.base_period = period_steps or 1
        self.min_lr = min_lr
        self.lr_shrink = lr_shrink
        self.shrink_min = shrink_min
        self.t_mult = t_mult

        self._period = self.base_period
        self._cycle = 0
        self._start_step = 0

        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        step = max(self.last_step, 0)
        lrs = []

        for i, base_lr in enumerate(self.base_lrs):
            cycle = self._cycle
            period = self._period
            start_step = self._start_step

            # If step exceeds current cycle, advance to next cycle
            while step >= start_step + 2 * period:
                start_step += 2 * period
                period = int(period * self.t_mult)
                cycle += 1

            # Save updated state
            self._cycle = cycle
            self._period = period
            self._start_step = start_step

            # Compute cycle-local variables
            t_curr = step - start_step
            max_lr = base_lr * (self.lr_shrink ** cycle)
            min_lr = self.min_lr * (self.lr_shrink ** cycle) if self.shrink_min else self.min_lr
            x = abs(t_curr / period - 1)
            lr = min_lr + (max_lr - min_lr) * max(0.0, 1 - x)
            lrs.append(lr)

        return lrs
