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
from pathlib import Path
from models import WrappedMode
from tasks.base_task import PRLTask
from utils import lazy_import_module, get_attr_from_cfg


class DREAMERTask(PRLTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.ckpt = get_attr_from_cfg(self.cfg, "model.upstream.ckpt")
        self.hook_layer_strs = get_attr_from_cfg(self.cfg, "model.upstream.expert.hooks.module_path", ["self.model.encoder.final_fused_layer"])
        self.hook_transform = get_attr_from_cfg(self.cfg, "model.upstream.expert.hooks.transform", "lambda input, output: output")

    
    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.train_fpath, split='train')
        return self.train_dataset

    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.train_fpath, split='test')
        return self.dev_dataset
    
    def get_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = self.build_dataset(self.train_fpath, split='test')
        return self.test_dataset

    def build_dataset(self, filename: str, clip_length: int = 4, split: str = "train"):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        # transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(Path(self.dataset_root) / filename, clip_length=clip_length, split=split)

    def build_model(self):
        UpstreamCls = lazy_import_module('models.upstream', self.upstream_select)
        DownstreamCls = lazy_import_module('models.downstream', self.downstream_select)
        upstream = UpstreamCls(self.ckpt, self.hook_layer_strs, self.hook_transform)
        print(upstream)
        downstream = DownstreamCls(self.downstream_classes)
        print(downstream)
        return WrappedMode(upstream, downstream)
        
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