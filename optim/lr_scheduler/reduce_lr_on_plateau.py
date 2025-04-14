#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : reduce_lr_on_plateau.py
@Time    : 2025/04/10 21:20:30
@Desc    : 
"""

import torch
from typing import Optional, List
from .base_lr_scheduler import BaseLRScheduler


class ReduceLROnPlateauScheduler(BaseLRScheduler):
    """
    Reduce learning rate when a metric has stopped improving, with optional warmup phase.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        niter_per_epoch: int,
        metric: Optional[str] = None,
        patience_epochs: int = 10,
        patience_steps: Optional[int] = None,
        factor: float = 0.1,
        threshold: float = 1e-4,
        mode: str = "min",
        warmup_epochs: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        warmup_start_lr: float = 0.0,
        last_step: int = -1,
        min_lr: float = 0.0,
    ):
        """
        Initialize the ReduceLROnPlateauScheduler.

        Args:
            optimizer (Optimizer): The optimizer to be scheduled.
            niter_per_epoch (int): Number of steps per epoch.
            metric (str, optional): The metric to be used for scheduling. Default: None.
            patience_epochs (int, optional): Number of epochs with no improvement after which learning rate will be reduced.
            patience_steps (int, optional): Number of steps with no improvement after which learning rate will be reduced.
            factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
            threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
            mode (str): One of `min`, `max`. In `min` mode, lr will be reduced when the metric stops decreasing.
                        In `max` mode, it will be reduced when the metric stops increasing.
            warmup_epochs (int, optional): Number of warmup epochs. Default: None.
            warmup_steps (int, optional): Number of warmup steps. Default: None.
            warmup_start_lr (float): Initial learning rate during warmup. Default: 0.0.
            last_step (int): The index of the last step. Default: -1.
            min_lr (float): Minimum learning rate. Default: 0.0.
        """
        if warmup_steps is None and warmup_epochs is not None:
            warmup_steps = warmup_epochs * niter_per_epoch
        if patience_steps is None and patience_epochs is not None:
            patience_steps = patience_epochs * niter_per_epoch

        self.warmup_steps = warmup_steps or 0
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr

        # Initialize PyTorch's ReduceLROnPlateau
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience_steps,
            factor=factor,
            mode=mode,
            threshold=threshold,
            min_lr=min_lr,
        )
        super().__init__(optimizer, last_step, metric)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for the current step.

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
                # Plateau phase: use the current learning rate
                lr = max(base_lr, self.min_lr)
            lrs.append(lr)

        return lrs

    def step(self, metrics: Optional[float] = None, step: Optional[int] = None):
        """
        Update the learning rate based on the given metric.

        Args:
            metrics (float, optional): The metric to monitor for plateau detection.
            step (int, optional): The current step. If None, it will increment the internal step counter.
        """
        # Use self.last_step + 1 if step is None
        if step is None:
            step = self.last_step + 1
            
        # Warmup phase
        if step < self.warmup_steps:
            super().step(metrics, step)
            return

        # Plateau detection
        if metrics is not None:
            self.lr_scheduler.step(metrics)
        else:
            super().step(metrics, step)