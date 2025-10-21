#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : tri_stage_lr_scheduler.py
@Time    : 2025/04/11 13:51:26
@Desc    : 
"""
import math
from typing import Optional, List, Tuple
from tyee.optim.lr_scheduler import BaseLRScheduler


class TriStageLRScheduler(BaseLRScheduler):
    """
    Tri-Stage Learning Rate Scheduler.

    Implements a learning rate schedule with three stages:
    1. Warmup: Linearly increase the learning rate from `init_lr` to `peak_lr`.
    2. Hold: Keep the learning rate constant at `peak_lr`.
    3. Decay: Exponentially decay the learning rate to `final_lr`.

    During warmup:
        lr = init_lr + (peak_lr - init_lr) * step / warmup_steps

    During hold:
        lr = peak_lr

    During decay:
        lr = peak_lr * exp(-decay_factor * (step - warmup_steps - hold_steps))

    After decay:
        lr = final_lr
    """

    def __init__(
        self,
        optimizer,
        niter_per_epoch: int,
        warmup_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        hold_epochs: Optional[int] = None,
        hold_steps: Optional[int] = None,
        decay_epochs: Optional[int] = None,
        decay_steps: Optional[int] = None,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.01,
        last_step: int = -1,
    ):
        """
        Initialize the TriStageLRScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of steps per epoch.
            warmup_epochs (int, optional): Number of warmup epochs. Default: None.
            warmup_steps (int, optional): Number of warmup steps. Default: None.
            hold_epochs (int, optional): Number of hold epochs. Default: None.
            hold_steps (int, optional): Number of hold steps. Default: None.
            decay_epochs (int, optional): Number of decay epochs. Default: None.
            decay_steps (int, optional): Number of decay steps. Default: None.
            init_lr_scale (float): Initial learning rate scale during warmup. Default: 0.01.
            final_lr_scale (float): Final learning rate scale after decay. Default: 0.01.
            last_step (int): The index of the last step. Default: -1.
        """
        # Convert epochs to steps if necessary
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch
        if hold_steps is None and hold_epochs is not None:
            hold_steps = hold_epochs * niter_per_epoch
        if decay_steps is None and decay_epochs is not None:
            decay_steps = decay_epochs * niter_per_epoch

        self.niter_per_epoch = niter_per_epoch
        self.warmup_steps = warmup_steps or 0
        self.hold_steps = hold_steps or 0
        self.decay_steps = decay_steps or 0

        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        # Compute decay factor for exponential decay
        self.decay_factor = -math.log(final_lr_scale) / max(self.decay_steps, 1)

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
            init_lr = self.init_lr_scale * base_lr
            peak_lr = base_lr
            final_lr = self.final_lr_scale * base_lr

            if step <= self.warmup_steps:
                # Warmup phase: linearly increase learning rate
                lr = init_lr + (peak_lr - init_lr) * step / self.warmup_steps
            elif step <= self.warmup_steps + self.hold_steps:
                # Hold phase: keep learning rate constant
                lr = peak_lr
            elif step <= self.warmup_steps + self.hold_steps + self.decay_steps:
                # Decay phase: exponentially decay learning rate
                decay_step = step - self.warmup_steps - self.hold_steps
                lr = peak_lr * math.exp(-self.decay_factor * decay_step)
            else:
                # After decay: keep learning rate constant at final_lr
                lr = final_lr

            lrs.append(lr)

        return lrs