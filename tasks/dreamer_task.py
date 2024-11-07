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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train_step(self, model, x, target, *args, **kwargs):
        output, _ = model(x, *args, **kwargs)
        loss = self.loss(output,target)
        return {
            'loss':loss
        }

    @torch.no_grad()
    def valid_step(self, model, x, target, *args, **kwargs):
        output, _ = model(x)
        loss = self.loss(output, target)

        return {
            'loss':loss
        }