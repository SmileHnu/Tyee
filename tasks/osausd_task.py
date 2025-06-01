#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : osausd_task.py
@Time    : 2025/05/24 16:27:28
@Desc    : 
"""


import torch
from tasks import PRLTask
import numpy as np
from utils import get_nested_field, lazy_import_module

class OSAUSDTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.imbalanced_ratio = 1.0
    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        ecg = sample['ecg'].float()
        spo2 = sample['spo2'].float()
        # print(ecg.shape, spo2.shape)
        label = sample['anomaly']
        # print(label)
        pred = model(ecg, spo2)
        # print(pred.shape, label.shape)
        loss = self.loss(pred, label, self.imbalanced_ratio)
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        ecg = sample['ecg'].float()
        spo2 = sample['spo2'].float()
        # print(x.shape)
        label = sample['anomaly']
        pred = model(ecg, spo2)
        loss = self.loss(pred, label, self.imbalanced_ratio)
        pred = (pred.data > 0).to(torch.float32)
        pred = pred.reshape(-1)
        label = (label > 0).to(torch.float32)
        label = label.reshape(-1)

        return{
            'loss': loss,
            'output': pred,
            'label': label
        }
    
    def on_train_start(self, trainer, *args, **kwargs):
        train_labels = []
        for sample in trainer.train_loader:
            labels = sample["anomaly"].numpy()
            train_labels.append(labels.flatten())
        train_labels = np.concatenate(train_labels)
        positive_count = np.sum(train_labels > 0)
        negative_count = np.sum(train_labels <= 0)
        self.imbalanced_ratio = negative_count / positive_count
        print(f"Imbalanced ratio: {self.imbalanced_ratio}")
            