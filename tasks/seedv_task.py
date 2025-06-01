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
        # print(model)
        # checkpoint = torch.load('/home/lingyus/code/G2G/G2G/logs/SEED5/2025-05-28-02:31:58/model_best.pth.tar', map_location='cpu')
        # model.load_state_dict(checkpoint['enc_module_state_dict'], strict=True)
        # model.rand_order = torch.tensor([
        #         [ 3, 17, 22, 59, 32, 24, 19,  7, 41, 30, 47, 44, 49, 36, 42, 45,  4, 55,
        #         10, 58, 52, 33, 18, 31, 61, 51,  9, 57, 25,  6, 37, 60, 28, 56,  5, 26,
        #         27, 12,  0, 53, 15, 16, 35, 23, 40, 38, 14, 50, 54, 20, 39, 13,  1, 46,
        #         8, 29, 48, 34, 43,  2, 21, 11],
        #         [58, 41, 51, 12, 14,  2, 34, 56,  6, 45, 23, 30, 38, 32, 31, 24, 43, 22,
        #         9, 54,  3, 19,  4, 20,  5, 27, 55, 17, 48, 40, 37, 52, 25, 46, 50, 26,
        #         0, 53, 18, 15,  7, 21, 11, 16, 35, 42,  1, 44, 10, 36, 60, 59, 39, 61,
        #         28, 47, 13, 49, 29, 57, 33,  8]
        #     ])
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