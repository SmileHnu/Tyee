#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : one_cycle_lr_scheduler.py
@Time    : 2025/05/10 16:37:27
@Desc    : 
"""


import torch
from typing import Optional, List
from tyee.optim.lr_scheduler import BaseLRScheduler


class OneCycleScheduler(BaseLRScheduler):
    """
    One Cycle learning rate scheduler with optional warmup phase.
    Wraps PyTorch's OneCycleLR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        niter_per_epoch: int,
        max_lr: float,
        epochs: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_step: int = -1,
        **kwargs
    ):
        """
        Args:
            optimizer (Optimizer): optimizer
            niter_per_epoch (int): number of steps per epoch
            max_lr (float): maximum learning rate
            epochs (int): total number of epochs
            pct_start (float): proportion of the cycle for increasing learning rate
            anneal_strategy (str): 'cos' or 'linear'
            div_factor (float): max_lr/initial_lr
            final_div_factor (float): min_lr=initial_lr/final_div_factor
            three_phase (bool): whether to use three phases
            last_step (int): previous step
            kwargs: other parameters
        """
        self.niter_per_epoch = niter_per_epoch
        self.epochs = epochs
        self.max_lr = max_lr

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=epochs * niter_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            three_phase=three_phase,
            last_epoch=last_step,
            **kwargs
        )
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for the current step
        """
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics: Optional[float] = None, step: Optional[int] = None):
        """
        Update learning rate
        """
        if step is not None:
            self.lr_scheduler.last_epoch = step
        if self.lr_scheduler.last_epoch + 1 < self.lr_scheduler.total_steps:
            self.lr_scheduler.step()
            self.last_step = step