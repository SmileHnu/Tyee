#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : emotion_recognition_task.py
@Time    : 2024/11/11 18:58:58
@Desc    : 
"""
import os
from . import PRLTask
from utils import lazy_import_module
from models import WrappedMode

class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg):
        
        super().__init__(cfg)
    

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.train_fpath, split='train')
        return self.train_dataset

    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.train_fpath, split='dev')
        return self.dev_dataset
    
    def get_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = self.build_dataset(self.train_fpath, split='test')
        return self.test_dataset

    def build_dataset(self, filename: str, split: str = "train"):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        # transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(self.dataset_root, filename), split=split)
    
    def build_model(self):
        up_module = lazy_import_module('models.upstream', self.upstream_select)
        upstream = up_module()

        down_module = lazy_import_module('models.downstream',self.downstream_select)
        downstream = down_module(classes=self.downstream_classes)

        return WrappedMode(upstream, downstream, self.upstream_trainable)

    def train_step(self, model, sample):
        # print(type(data),type(target))
        x = sample["x"]
        target = sample["target"]
        output= model(x)
        # print(target.device)
        # print(self.loss.weight.device)
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output,target)

        return {
            'loss':loss,
            'output':output,
            'target':target
        }
        
    def valid_step(self, model, sample):
        # print(type(data),type(target))
        x = sample["x"]
        target = sample["target"]
        output= model(x)
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output, target)

        return {
            'loss':loss,
            'output':output,
            'target':target
        }
