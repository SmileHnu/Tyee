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

from tasks import PRLTask
from utils import get_nested_field

class SleepEDFxTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.checkpoint = get_nested_field(cfg, 'model.upstream.checkpoint', default=None)

    def build_dataset(self, filename):
        pass

    def get_train_dataset(self):
        pass
    
    def get_dev_dataset(self):
        pass

    def get_test_dataset(self):
        pass

    def build_model(self):
        pass

    def train_step(self, model, x, *args, **kwargs):
        pass

    def eval_step(self, model, x, *args, **kwargs):
        pass