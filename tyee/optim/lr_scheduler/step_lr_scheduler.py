#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : step_lr_scheduler.py
@Time    : 2025/04/10 21:03:08
@Desc    : 
"""


import math
from typing import Optional, List
from tyee.optim.lr_scheduler import BaseLRScheduler


class StepLRScheduler(BaseLRScheduler):
    """
    Step Learning Rate Scheduler.

    This scheduler adjusts the learning rate at fixed intervals (steps or epochs)
    by multiplying it with a decay factor (`gamma`). It also supports an optional
    warmup phase where the learning rate increases linearly from `warmup_start_lr`
    to the base learning rate.

    During warmup::

        lr = warmup_start_lr + (base_lr - warmup_start_lr) * step / warmup_steps

    After warmup::

        decay_steps = (step - self.warmup_steps) // self.step_size
        lr = base_lr * (self.gamma ** decay_steps)
    """

    def __init__(
        self,
        optimizer,
        niter_per_epoch: int,
        step_size: Optional[int] = None,
        epoch_size: Optional[int] = None,
        gamma: float = 0.1,
        warmup_steps: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        last_step: int = -1,
    ):
        """
        Initialize the StepLRScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of iterations per epoch.
            step_size (int, optional): Number of steps between each learning rate decay.
            epoch_size (int, optional): Number of epochs between each learning rate decay.
            gamma (float): Decay factor for learning rate adjustment. Default: 0.1.
            warmup_steps Optional[int] = None: Number of warmup steps. Default: 0.
            warmup_epochs Optional[int] = None: Number of warmup epochs. Default: 0.
            warmup_start_lr (float): Initial learning rate during warmup. Default: 0.0.
            last_step (int): The index of the last step. Default: -1.
        """
        if step_size is None and epoch_size is not None:
            step_size = epoch_size * niter_per_epoch
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch

        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps or 0
        self.warmup_start_lr = warmup_start_lr
        self.niter_per_epoch = niter_per_epoch

        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """
        Compute the learning rate for each parameter group.

        Returns:
            List[float]: The learning rate for each parameter group.
        """
        step = self.last_step
        lrs = []

        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Warmup phase: linearly increase learning rate
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
            else:
                # Decay phase: adjust learning rate at fixed intervals
                decay_steps = (step - self.warmup_steps) // self.step_size
                lr = base_lr * (self.gamma ** decay_steps)

            lrs.append(lr)

        return lrs