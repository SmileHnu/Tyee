#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : inverse_square_root_scheduler.py
@Time    : 2025/04/11 13:14:58
@Desc    : 
"""

import math
from typing import Optional, List
from tyee.optim.lr_scheduler import BaseLRScheduler


class InverseSquareRootScheduler(BaseLRScheduler):
    """
    Decay the learning rate based on the inverse square root of the update number.

    Supports a warmup phase where the learning rate linearly increases from
    `warmup_start_lr` to the base learning rate (`base_lr`). After warmup, the
    learning rate decays proportionally to the inverse square root of the update number.

    During warmup:
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * step / warmup_steps

    After warmup:
        decay_factor = base_lr * sqrt(warmup_steps)
        lr = decay_factor / sqrt(step)
    """

    def __init__(
        self,
        optimizer,
        niter_per_epoch: int,
        warmup_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        last_step: int = -1,
    ):
        """
        Initialize the InverseSquareRootScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of steps per epoch.
            warmup_epochs (int, optional): Number of warmup epochs. Default: None.
            warmup_steps (int, optional): Number of warmup steps. Default: None.
            warmup_start_lr (float): Initial learning rate during warmup. Default: 0.0.
            last_step (int): The index of the last step. Default: -1.
        """
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch

        self.warmup_steps = warmup_steps or 0
        self.warmup_start_lr = warmup_start_lr

        # Compute the decay factor for the inverse square root phase
        self.decay_factor = None  # Will be initialized after warmup
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """
        Compute the learning rate for each parameter group.

        Returns:
            List[float]: The learning rate for each parameter group.
        """
        step = max(self.last_step, 1)  # Avoid division by zero
        lrs = []

        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                # Warmup phase: linearly increase learning rate
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
            else:
                # Inverse square root decay phase
                if self.decay_factor is None:
                    self.decay_factor = base_lr * math.sqrt(self.warmup_steps)
                lr = self.decay_factor / math.sqrt(step)
            lrs.append(lr)

        return lrs