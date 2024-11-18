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


from . import PRLTask
from utils import lazy_import_module
from models import WrappedMode

class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg):
        
        super().__init__(cfg)
        
    def build_model(self):
        up_module = lazy_import_module('models.upstream', self.upstream_select)
        upstream = up_module()

        down_module = lazy_import_module('models.downstream',self.downstream_select)
        downstream = down_module(classes=self.downstream_classes)

        return WrappedMode(upstream, downstream, self.upstream_trainable)

    def train_step(self, model, data, target):
        # print(type(data),type(target))
        
        output= model(data)
        # print(target.device)
        # print(self.loss.weight.device)
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output,target)

        return {
            'loss':loss
        }
        
    def valid_step(self, model, data, target):
        # print(type(data),type(target))
        output= model(data)
        self.loss.weight = self.loss.weight.to(target.device)
        loss = self.loss(output, target)

        return {
            'loss':loss,
            'output':output,
            'target':target
        }
