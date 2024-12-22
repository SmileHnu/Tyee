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
from dataset import DatasetType
import torch.nn.functional as F
from utils import get_nested_field, lazy_import_module

class BCICIV2ATask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.checkpoint = get_nested_field(cfg, 'model.upstream.checkpoint', default=None)
        
        self.test_subject = get_nested_field(cfg, 'dataset.test_subject', None)
        

    def build_dataset(self, sub, path, split):
        Dataset = lazy_import_module('dataset', self.dataset)
        return Dataset(sub, path, is_few_EA=True, target_sample=1024, split=split)

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.test_subject, self.dataset_root, DatasetType.TRAIN)
        return self.train_dataset
    
    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.test_subject, self.dataset_root, DatasetType.TEST)
        return self.dev_dataset

    def get_test_dataset(self):
        return None
    def build_model(self):
        model = lazy_import_module('models.upstream', self.upstream_select)
        return model(load_path = self.checkpoint)
    
    def build_optimizer(self, model: torch.nn.Module):
        return model.optimizer

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['x']
        target = sample['target']
        x, pred = model(x)
        
        loss = self.loss(pred, target)
        
        return{
            'loss': loss,
            'output': pred,
            'target': target
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['x']
        target = sample['target']
        x, pred = model(x)
        
        loss = self.loss(pred, target)
        
        return{
            'loss': loss,
            'output': pred,
            'target': target
        }