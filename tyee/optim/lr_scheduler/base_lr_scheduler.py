#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : base_lr_scheduler.py
@Time    : 2025/04/10 16:26:28
@Desc    : 
"""

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class BaseLRScheduler(LRScheduler):
    """
    Base class for learning rate schedulers.

    This class provides a foundation for implementing custom learning rate
    schedulers. Subclasses must implement the `get_lr` method.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        last_step: int = -1,
        metric: Optional[str] = None,
        metric_source: Optional[str] = None,
    ):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be scheduled.
            last_step (int): The index of the last step. Default: -1.
            metric (str, optional): The metric to be used for scheduling. Default: None.
        """
        self.optimizer = optimizer
        self.last_step = last_step
        self.metric_source = metric_source
        self.metric = metric
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self) -> List[float]:
        """
        Get the learning rate for the current step.

        Returns:
            List[float]: The learning rate for each parameter group.
        """
        raise NotImplementedError("get_lr() must be implemented in subclasses.")

    def step(
        self,
        metrics: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Update the learning rate for the current step.

        Args:
            metrics (float, optional): The metric to be used for scheduling. Default: None.
            step (int, optional): The index of the current step. Default: None.
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        super().step(step)