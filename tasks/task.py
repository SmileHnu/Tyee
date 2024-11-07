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


        # 数据集
        self.train_dataset = self.__build_dataset(self.train_fpath)
        self.dev_dataset = self.__build_dataset(self.eval_fpath[0])
        self.test_dataset = self.__build_dataset(self.eval_fpath[1])
        
        # 模型、优化器与学习率调度器
        self.model = self.__build_model()
        self.optimizer = self.__build_optimizer()
        self.lr_scheduler = self.__build_lr_scheduler()
        self.loss = self.__build_loss()

    
    def __build_dataset(self, filename: str):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(self.dataset_root, filename), transforms)

    def __build_model(self):
        raise NotImplementedError

    def __build_loss(self):
        """ 构建损失函数 """
        loss_cls = lazy_import_module('torch.nn', self.loss_select)
        loss_args = {'weight': torch.tensor(self.loss_weight, dtype=torch.float32)} if self.loss_weight else {}
        return loss_cls(**loss_args)

    def __build_optimizer(self):
        """ 构建优化器 """
        optimizer_cls = lazy_import_module('torch.optim', self.optimizer_select)
        return optimizer_cls(self.model.parameters(), lr=self.lr)

    def __build_lr_scheduler(self):
        """ 构建学习率调度器 """
        scheduler_cls = lazy_import_module('torch.optim.lr_scheduler', self.lr_scheduler_select)
        return scheduler_cls(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def __get_dataset(self, attr_name, fpath: str) -> Dataset:
        """
        通用的获取 dataset 方法，如果已经存在则直接返回，否则构建并返回
        :param attr_name: 用于标识 dataset 的属性名称
        :param fpath: 数据集文件子路径
        :return: dataset
        """
        dataset = getattr(self, attr_name, None)
        if dataset is None:
            dataset = self.__build_dataset(fpath)
            setattr(self, attr_name, dataset)
        return dataset
    
    @property
    def model(self):
        if self.model is None:
            self.model = self.__build_model()
        return self.model

    @property
    def train_dataset(self) -> Dataset:
        return self.__get_dataset("train_dataset", self.train_fpath)
    
    @property
    def dev_dataset(self) -> Dataset:
        return self.__get_dataset("dev_dataset", self.eval_fpath[0])
    
    @property
    def test_dataset(self) -> Dataset:
        return self.__get_dataset("test_dataset", self.eval_fpath[1])

    @property
    def optimizer(self):
        if self.optimizer is None:
            self.optimizer = self.__build_optimizer()
        return self.optimizer

    @property
    def lr_scheduler(self):
        if self.lr_scheduler is None:
            self.lr_scheduler = self.__build_lr_scheduler()
        return self.lr_scheduler
  
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self):
        return {
            "args": self.cfg,
            "model": self.model.state_dict(),
            "optimzer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def save_checkpoint(self, filename):
        checkpoint = self.state_dict()
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        ckpt_params = torch.load(filename)
        self.args = ckpt_params["args"]
        self.model.load_state_dict(ckpt_params["model"])
        self.optimizer.load_state_dict(ckpt_params["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt_params["lr_scheduler"])

    def train_step(self, model, x, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def valid_step(self, model, x, *args, **kwargs):
        raise NotImplementedError