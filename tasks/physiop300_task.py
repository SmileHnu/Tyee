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

class PhysioP300Task(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.checkpoint = get_nested_field(cfg, 'model.upstream.checkpoint', default=None)
        
        self.train_subjects = get_nested_field(cfg, 'dataset.train_subjects', None)
        self.val_subjects = get_nested_field(cfg, 'dataset.val_subjects', None)
        

    def build_dataset(self, path, subjects=None):
        Dataset = lazy_import_module('dataset', self.dataset)
        return Dataset(path, subjects)

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.dataset_root, subjects=self.train_subjects)
        return self.train_dataset
    
    def get_val_dataset(self):
        if self.val_dataset is None:
            self.val_dataset = self.build_dataset(self.dataset_root, subjects=self.val_subjects)
        return self.val_dataset

    def get_test_dataset(self):
        return None

    def build_model(self):
        model = lazy_import_module('models.upstream', self.upstream_select)
        return model(load_path = self.checkpoint)
    
    def build_optimizer(self, model: torch.nn.Module):
        return model.optimizer

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['x']
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
        x = sample['x']
        label = sample['label']
        x, pred = model(x)
        loss = self.loss(pred, label)
        pred =  torch.softmax(pred, dim=-1)[:,1]
        return{
            'loss': loss,
            'output': pred,
            'label': label
        }