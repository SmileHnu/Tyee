#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : dreamer_task.py
@Time    : 2024/11/06 14:51:19
@Desc    : 
"""
import torch
from . import PRLTask


class DREAMETask(PRLTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self, x, target, *args, **kwargs):
        x, target = x.to(self.rank), target.to(self.rank)
        output, _ = self.model(x, x.size(1))
        loss = self.loss(output,target)
        return {
            'loss':loss
        }

    @torch.no_grad()
    def valid_step(self, x, target, *args, **kwargs):
        output, _ = self.model(x)
        loss = self.loss(output, target)

        return {
            'loss':loss
        }