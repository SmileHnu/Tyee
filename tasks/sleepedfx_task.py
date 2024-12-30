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

        self.checkpoint = get_nested_field(cfg, 'model.upstream.checkpoint', default=None)
        
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

        self.dev_subjects = get_nested_field(cfg, 'dataset.dev_subjects', None)
        self.train_subjects = [subject for subject in self.all_subjects if subject not in self.dev_subjects]
        print(f"Train subjects: {self.train_subjects}")
        print(f"Dev subjects: {self.dev_subjects}")

    def build_dataset(self, path, subjects=None):
        Dataset = lazy_import_module('dataset', self.dataset)
        return Dataset(path, subjects)

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.dataset_root, subjects=self.train_subjects)
        return self.train_dataset
    
    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.dataset_root, subjects=self.dev_subjects)
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