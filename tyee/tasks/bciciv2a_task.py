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
from torch import nn
from pathlib import Path
from tasks.base_task import PRLTask
from utils import lazy_import_module, get_nested_field


class BCICIV2aTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model_select = get_nested_field(cfg, 'model.select', '')
        self.model_params = get_nested_field(cfg, 'model', {})
        # print(self.model_params)
    
    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        return model
        
    def train_step(self, model: nn.Module, sample: dict[str, torch.Tensor]):
        x = sample["eeg"].float()
        # print(x.shape)
        label = sample["event"]
        # x = x.unsqueeze(1)
        pred = model(x)
        # print(pred.shape)
        # print(label.shape)
        loss = self.loss(pred, label)
        return {
            "output": pred,
            "label": label,
            "loss": loss
        }

    @torch.no_grad()
    def valid_step(self, model, sample: dict[str, torch.Tensor]):
        x = sample["eeg"].float()
        # print(x.shape)
        label = sample["event"]
        # x = x.unsqueeze(1)
        pred = model(x)
        loss = self.loss(pred, label)

        return {
            "output": pred,
            "label": label,
            "loss": loss
        }