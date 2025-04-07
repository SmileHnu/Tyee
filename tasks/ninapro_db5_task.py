#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : ninapro_db5_task.py
@Time    : 2025/03/28 10:28:32
@Desc    : 
"""

import torch
from tasks import PRLTask
from utils import get_nested_field, lazy_import_module

class NinaproDB5Task(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.model_select = get_nested_field(cfg, 'model.select', '')
        self.model_params = get_nested_field(cfg, 'model', {})


    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        # print(model)
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['emg']['signals']
        # 将信号沿着通道维度拼接起来
        x = x.float()
        # print(x.shape)
        label = sample['label']
        pred = model(x)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x= sample['emg']['signals']
        x = x.float()
        # print(x.shape)
        label = sample['label']
        pred = model(x)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }