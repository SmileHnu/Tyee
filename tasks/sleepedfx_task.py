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

        self.all_subjects = [0, 2, 4, 5, 6, 7,
                              8, 9, 11, 12, 13, 14,
                              15, 16, 17, 18, 19, 21,
                              22, 23, 24, 25, 26, 29, 
                              30, 31, 32, 33, 34, 35, 
                              37, 38, 40, 42, 44, 45, 
                              46, 47, 48, 49, 51, 52, 
                              53, 54, 55, 56, 57, 58, 
                              59, 61, 62, 63, 64, 65, 
                              66, 71, 72, 73, 74, 75, 
                              76, 77, 81, 82]
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
        eeg = sample['eeg']['signals']
        eog = sample['eog']['signals']
        emg = sample['emg']['signals']
        rsp = sample['rsp']['signals']
        # 将信号沿着通道维度拼接起来
        x = torch.cat((eeg, eog, emg, rsp), dim=1)
        # x = eeg
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
        eeg = sample['eeg']['signals']
        eog = sample['eog']['signals']
        emg = sample['emg']['signals']
        rsp = sample['rsp']['signals']
        # 将信号沿着通道维度拼接起来
        x = torch.cat((eeg, eog, emg, rsp), dim=1)
        # x = eeg
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