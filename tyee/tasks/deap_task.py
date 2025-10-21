#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : deap_task.py
@Time    : 2025/05/25 20:48:30
@Desc    : 
"""


import torch
from tyee.tasks import BaseTask
from tyee.utils import get_nested_field, lazy_import_module

class DEAPTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        print(model)
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['mulit4'].float()
        # print(x.shape)
        label = sample['arousal']
        pred = model(x)
        
        loss = self.loss(pred, label)
        # softmax = torch.nn.Softmax(dim=1)
        # pred = softmax(pred)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x= sample['mulit4'].float()
        # print(x)
        # print(x.shape)
        label = sample['arousal']
        pred = model(x)
        # print(pred.shape, label.shape)
        # print(f'valid:{x}')
        # label = torch.argmax(label, dim=1)
        # print(label)
        loss = self.loss(pred, label)
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        # pred_true = torch.argmax(pred, dim=1)
        # print(pred_true, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }