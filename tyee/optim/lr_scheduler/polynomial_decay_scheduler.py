#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : polynomial_decay_scheduler.py
@Time    : 2025/04/11 13:41:48
@Desc    : 
"""

import math
from typing import Optional, List
from .base_lr_scheduler import BaseLRScheduler


class PolynomialDecayLRScheduler(BaseLRScheduler):
    """
    Decay the learning rate based on a polynomial schedule.

    Supports a warmup phase where the learning rate linearly increases from
    `warmup_start_lr` to the base learning rate (`base_lr`). After warmup, the
    learning rate decays polynomially to the `end_learning_rate` over the
    total number of steps.

    During warmup:
        lr = warmup_start_lr + (base_lr - warmup_start_lr) * step / warmup_steps

    After warmup:
        lr = (base_lr - end_learning_rate) * (1 - (step - warmup_steps) / (total_steps - warmup_steps))^power + end_learning_rate
    """

    def __init__(
        self,
        optimizer,
        niter_per_epoch: int,
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        end_learning_rate: float = 0.0,
        power: float = 1.0,
        last_step: int = -1,
    ):
        """
        Initialize the PolynomialDecayLRScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of steps per epoch.
            total_epochs (int, optional): Total number of epochs. Default: None.
            total_steps (int, optional): Total number of steps for cosine annealing. Default: None.
            warmup_epochs (Optional[int]): Number of warmup epochs. Default: None.
            warmup_steps (Optional[int]): Number of warmup steps. Default: None.
            warmup_start_lr (float): Initial learning rate during warmup. Default: 0.0.
            end_learning_rate (float): Final learning rate after decay. Default: 0.0.
            power (float): The power of the polynomial decay. Default: 1.0.
            last_step (int): The index of the last step. Default: -1.
        """
        if total_steps is None and total_epochs is not None:
            total_steps = total_epochs * niter_per_epoch
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch

        self.niter_per_epoch = niter_per_epoch
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps or 0
        self.warmup_start_lr = warmup_start_lr
        self.end_learning_rate = end_learning_rate
        self.power = power

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
                # Polynomial decay phase
                if self.total_steps is None or step >= self.total_steps:
                    lr = self.end_learning_rate
                else:
                    decay_steps = self.total_steps - self.warmup_steps
                    pct_remaining = 1 - (step - self.warmup_steps) / decay_steps
                    lr = (base_lr - self.end_learning_rate) * (pct_remaining ** self.power) + self.end_learning_rate
            lrs.append(lr)

        return lrs
