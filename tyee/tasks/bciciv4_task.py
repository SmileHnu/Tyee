#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv4_task.py
@Time    : 2025/03/28 21:40:58
@Desc    : 
"""


import torch
import scipy
from tyee.tasks import BaseTask
from torch import nn
import torch.nn.functional as F
from tyee.utils import get_nested_field, lazy_import_module

def correlation_metric(x, y):
    """
     Cosine distance calculation metric
    """
    cos_metric = nn.CosineSimilarity(dim=-1, eps=1e-08)

    cos_sim = torch.mean(cos_metric(x, y))

    return cos_sim

class BCICIV4Task(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.model_select = get_nested_field(cfg, 'model.select', '')
        self.model_params = get_nested_field(cfg, 'model', {})


    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        print(self.model_params)
        model = model_name(**self.model_params)
        # print(model)
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['ecog']
        x = x.float()
        # print(x)
        label = sample['dg'].float()
        # print(label)
        pred = model(x)
        # print(pred)
        # loss = self.loss(pred, label)
        loss = F.mse_loss(pred, label)
        corr = correlation_metric(pred, label)
        return{
            'loss': 0.5*loss + 0.5*(1. - corr),
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['ecog']
        x = x.float()
        SIZE = 64
        bounds = x.shape[-1] // SIZE * SIZE
        x = x[..., :bounds]
        # print(x.shape)
        label = sample['dg'].float()
        label = label[...,:bounds]
        pred = model(x)
        print(label.shape, pred.shape)
        loss = F.mse_loss(pred, label)
        corr = correlation_metric(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }