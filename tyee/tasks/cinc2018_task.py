#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : cinc2018_task.py
@Time    : 2025/05/31 13:04:39
@Desc    : 
"""


import torch
from tasks import PRLTask
from utils import get_nested_field, lazy_import_module

class CinC2018Task(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        # print(model)
        return model
    def set_optimizer_params(self, model, lr, layer_decay, weight_decay = 0.00001):
        params = (list(model.linear.parameters()) )
        param_groups = [
            {'params': params, 'lr': lr, 'weight_decay': weight_decay},
        ]
        return param_groups

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        ss = sample['ss'].float()
        resp = sample['resp'].float()
        ecg = sample['ecg'].float()
        label = sample['stage']
        pred = model(ss, ecg, resp)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        ss = sample['ss'].float()
        resp = sample['resp'].float()
        ecg = sample['ecg'].float()
        label = sample['stage']
        pred = model(ss, ecg, resp)
        loss = self.loss(pred, label)
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }