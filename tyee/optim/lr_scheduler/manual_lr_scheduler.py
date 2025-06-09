#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : manual_lr_scheduler.py
@Time    : 2025/04/11 13:25:25
@Desc    : 
"""

from typing import Optional, Dict, List
from .base_lr_scheduler import BaseLRScheduler


class ManualScheduler(BaseLRScheduler):
    """
    Manually set the learning rate based on a predefined schedule for epochs or steps.

    Supports setting learning rates for specific epochs and steps. Epochs are converted
    to steps during initialization based on the number of steps per epoch.
    """

    def __init__(
        self,
        optimizer,
        niter_per_epoch: int,
        epoch2lr: Optional[Dict[int, float]] = None,
        step2lr: Optional[Dict[int, float]] = None,
        last_step: int = -1,
    ):
        """
        Initialize the ManualScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of steps per epoch.
            epoch2lr (dict, optional): A dictionary mapping epochs to learning rates.
            step2lr (dict, optional): A dictionary mapping steps to learning rates.
            niter_per_epoch (int): Number of steps per epoch. Default: 1.
            last_step (int): The index of the last step. Default: -1.
        """
        self.niter_per_epoch = niter_per_epoch
        self.step2lr = step2lr or {}

        # Convert epoch2lr to step2lr
        if epoch2lr:
            for epoch, lr in epoch2lr.items():
                step = epoch * niter_per_epoch
                self.step2lr[step] = lr

        # Sort step2lr by step for easier lookup
        self.step2lr = dict(sorted(self.step2lr.items()))
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """
        Compute the learning rate for the current step.

        Returns:
            List[float]: The learning rate for each parameter group.
        """
        # Find the largest step <= current step in step2lr
        manual_keys = [k for k in self.step2lr if k <= self.last_step]
        if manual_keys:
            lr = self.step2lr[max(manual_keys)]
        else:
            # Default to the current learning rate if no match is found
            lr = self.optimizer.param_groups[0]["lr"]

        return [lr for _ in self.optimizer.param_groups]

        