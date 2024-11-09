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
import os
import torch
from torch.utils.data import Dataset
from utils import lazy_import_module, get_attr_from_cfg





class PRLTask(object):
    def __init__(self, cfg: dict) -> None:
        # 解析配置
        self.cfg = cfg
        
        # 数据集路径和变换
        self.dataset_root = get_attr_from_cfg(cfg, 'dataset.path', '')
        self.train_fpath = get_attr_from_cfg(cfg, 'dataset.train', '')
        self.eval_fpath = get_attr_from_cfg(cfg, 'dataset.eval', [])
        self.transforms_select = get_attr_from_cfg(cfg, 'dataset.transforms.select', [])
        self.dataset = get_attr_from_cfg(cfg, 'dataset.dataset', '')

        # 模型配置
        self.downstream_classes = get_attr_from_cfg(cfg, 'model.downstream.classes', 1)
        self.downstream_select = get_attr_from_cfg(cfg, 'model.downstream.select', '')
        self.upstream_select = get_attr_from_cfg(cfg, 'model.upstream.select', '')
        self.upstream_trainable = get_attr_from_cfg(cfg, 'model.upstream.trainable', False)

        # 损失函数配置
        self.loss_select = get_attr_from_cfg(cfg, 'task.loss.select', '')
        self.loss_weight = get_attr_from_cfg(cfg, 'task.loss.weight', [])

        # 优化器配置
        self.optimizer_select = get_attr_from_cfg(cfg, 'optimizer.select', '')
        self.lr = get_attr_from_cfg(cfg, 'optimizer.lr', 0.01)

        # 学习率调度器配置
        self.lr_scheduler_select = get_attr_from_cfg(cfg, 'lr_scheduler.select', '')
        self.step_size = get_attr_from_cfg(cfg, 'lr_scheduler.step_size', 20)
        self.gamma = get_attr_from_cfg(cfg, 'lr_scheduler.gamma', 0.1)


        self.loss = self.build_loss()

    
    def build_dataset(self, filename: str):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(self.dataset_root, filename), transforms)

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        """ 构建损失函数 """
        loss_cls = lazy_import_module('torch.nn', self.loss_select)
        loss_args = {'weight': torch.tensor(self.loss_weight, dtype=torch.float32)} if self.loss_weight else {}
        return loss_cls(**loss_args)

    def build_optimizer(self,model):
        """ 构建优化器 """
        optimizer_cls = lazy_import_module('torch.optim', self.optimizer_select)
        return optimizer_cls(model.parameters(), lr=self.lr)

    def build_lr_scheduler(self,optimizer):
        """ 构建学习率调度器 """
        scheduler_cls = lazy_import_module('torch.optim.lr_scheduler', self.lr_scheduler_select)
        return scheduler_cls(optimizer, step_size=self.step_size, gamma=self.gamma)

    # def get_dataset(self, attr_name, fpath: str) -> Dataset:
    #     """
    #     通用的获取 dataset 方法，如果已经存在则直接返回，否则构建并返回
    #     :param attr_name: 用于标识 dataset 的属性名称
    #     :param fpath: 数据集文件子路径
    #     :return: dataset
    #     """
    #     dataset = getattr(self, attr_name, None)
    #     if dataset is None:
    #         dataset = self.build_dataset(fpath)
    #         setattr(self, attr_name, dataset)
    #     return dataset
    
    # @property
    # def train_dataset(self) -> Dataset:
    #     return self.build_dataset(self.train_fpath)
    
    # @property
    # def dev_dataset(self) -> Dataset:
    #     return self.build_dataset(self.eval_fpath[0])
    
    # @property
    # def test_dataset(self) -> Dataset:
    #     return self.build_dataset(self.eval_fpath[1])

    
    def train_step(self, model, x, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def valid_step(self, model, x, *args, **kwargs):
        raise NotImplementedError