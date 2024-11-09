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
import os
import torch
from . import PRLTask
from utils import lazy_import_module, get_attr_from_cfg


class DREAMETask(PRLTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_dataset(self, filename: str, split="train"):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(self.dataset_root, filename), clip_length=4, split="train")

    def build_model(self):
        raise NotImplementedError
        
    def train_step(self, model, x, target, *args, **kwargs):
        output, _ = model(x, *args, **kwargs)
        loss = self.loss(output,target)
        return {
            'loss':loss
        }

    @torch.no_grad()
    def valid_step(self, model, x, target, *args, **kwargs):
        output, _ = model(x)
        loss = self.loss(output, target)

        return {
            'loss':loss
        }