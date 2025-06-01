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
from tasks import PRLTask
from utils import get_nested_field, lazy_import_module

class SleepEDFxTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        # print(model)
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        eeg = sample['eeg'].float()
        eog = sample['eog'].float()
        device = eeg.device
        label = sample['stage']
        pred = model(eeg, eog)
        label = label.reshape(-1)
        pred = pred.reshape(-1, pred.size(-1))
        self.loss.weight = self.loss.weight.to(device)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        eeg = sample['eeg'].float()
        eog = sample['eog'].float()
        device = eeg.device
        label = sample['stage']
        pred = model(eeg, eog)
        label = label.reshape(-1)
        pred = pred.reshape(-1, pred.size(-1))
        self.loss.weight = self.loss.weight.to(device)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }