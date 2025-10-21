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

class KaggleERNTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        return model

    def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float = 1e-5):
        params = (
            list(model.chan_conv.parameters()) +
            list(model.linear_probe1.parameters()) +
            list(model.linear_probe2.parameters())
        )
        param_groups = [
            {'params': params, 'lr': lr, 'weight_decay': weight_decay},
        ]
        return param_groups
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['eeg'].float()
        label = sample['label']
        x, pred = model(x)
        loss = self.loss(pred, label)
        pred =  torch.softmax(pred, dim=-1)[:,1]
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['eeg'].float()
        label = sample['label']
        x, pred = model(x)
        loss = self.loss(pred, label)
        pred =  torch.softmax(pred, dim=-1)[:,1]
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }