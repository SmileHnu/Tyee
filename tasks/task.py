#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : task.py
@Time    : 2024/09/25 16:41:26
@Desc    : 
"""
import abc

class PRLTask(object):
    def __init__(self) -> None:
        self.train_dataloader = None
        self.dev_dataloader = None
        self.test_dataloader = None
        self.model = None
        self.optimizer = None
        self.loss = None
        pass
    
    # 解析cfg构造dataloader
    @abc.abstractmethod
    def load_data(self,cfg):
        pass

    # 解析cfg获取model
    @abc.abstractmethod
    def load_model(self,cfg):
        pass

    # 解析cfg获取loss
    @abc.abstractmethod
    def load_loss(self,cfg):
        pass

    # 解析cfg获取optimizer
    @abc.abstractmethod
    def load_optimizer(self,cfg):
        pass

    # 给trainer传递dataloader
    @abc.abstractmethod
    def get_data(self, ):
        pass
    
    @abc.abstractmethod
    def train_step(self, ):
        pass

    @abc.abstractmethod
    def valid_step(self, ):
        pass
    
    # 给trainer传递optimizer
    @abc.abstractmethod
    def get_optimizer(self, ):
        pass
    
    @abc.abstractmethod
    def train(self,) -> None:
        pass
    
    @abc.abstractmethod
    def eval(self,) -> None:
        pass

    @abc.abstractmethod   
    def dynamic_import(self, module_name, class_name):
        pass
