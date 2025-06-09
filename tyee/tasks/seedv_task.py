#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : seedv_task.py
@Time    : 2025/03/24 19:43:10
@Desc    : 
"""

import torch
from tasks import PRLTask
from utils import get_nested_field, lazy_import_module

class SEEDVFeatureTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        return model
    
    def set_optimizer_params(self, model, lr, layer_decay, weight_decay = 0.00001):
        g2g_params = []
        backbone_params = []
        fc_params = []
        for pname, p in model.named_parameters():
                if "relation" in str(pname):
                    g2g_params += [p]
                elif "backbone" in str(pname):
                    backbone_params += [p]
                else:
                    fc_params += [p]
        
        param_groups = [{'params': g2g_params, 'lr': lr, 'weight_decay': weight_decay},
                        {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay},
                        {'params': fc_params, 'lr': lr , 'weight_decay': weight_decay}]
        return param_groups

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['eeg_eog']
        x = x.float()
        # print(x.shape)
        label = sample['emotion']
        pred = model(x)
        # print(pred.shape, label.shape)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['eeg_eog']
        x = x.float()
        # print(x.shape)
        label = sample['emotion']
        pred = model(x)
        loss = self.loss(pred, label)
        softmax = torch.nn.Softmax(dim=1)
        pred = softmax(pred)
        # print(loss)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }