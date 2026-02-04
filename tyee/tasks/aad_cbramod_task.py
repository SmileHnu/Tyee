#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : kaggleern_task.py
@Time    : 2024/12/17 20:43:09
@Desc    : 
"""

import torch
from tyee.tasks import BaseTask
from tyee.utils import get_nested_field, lazy_import_module
from einops import rearrange

class AADTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        return model

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        eeg = sample['eeg'].float()
        label = sample['label']
        eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg)
        loss = self.loss(pred, label.float())
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        eeg = sample['eeg'].float()
        label = sample['label']
        eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg)
        loss = self.loss(pred, label.float())
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }