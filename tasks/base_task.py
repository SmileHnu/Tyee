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
from utils import lazy_import_module, get_nested_field
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler




class PRLTask(object):
    def __init__(self, cfg: dict) -> None:
        # 解析配置
        self.cfg = cfg
        
        # 数据集路径和变换
        self.dataset_root = get_nested_field(cfg, 'dataset.path', '')
        self.train_fpath = get_nested_field(cfg, 'dataset.train', '')
        self.eval_fpath = get_nested_field(cfg, 'dataset.eval', [])
        self.transforms_select = get_nested_field(cfg, 'dataset.transforms.select', [])
        self.dataset = get_nested_field(cfg, 'dataset.dataset', '')

        # 模型配置
        self.downstream_classes = get_nested_field(cfg, 'model.downstream.classes', 1)
        self.downstream_select = get_nested_field(cfg, 'model.downstream.select', '')
        self.upstream_select = get_nested_field(cfg, 'model.upstream.select', '')
        self.upstream_trainable = get_nested_field(cfg, 'model.upstream.trainable', False)

        # 损失函数配置
        self.loss_select = get_nested_field(cfg, 'task.loss.select', '')
        self.loss_weight = get_nested_field(cfg, 'task.loss.weight', [])

        # 优化器配置
        self.optimizer_select = get_nested_field(cfg, 'optimizer.select', '')
        self.lr = get_nested_field(cfg, 'optimizer.lr', 0.0001)

        # 学习率调度器配置
        self.lr_scheduler_select = get_nested_field(cfg, 'lr_scheduler.select', '')
        self.step_size = get_nested_field(cfg, 'lr_scheduler.step_size', 20)
        self.gamma = get_nested_field(cfg, 'lr_scheduler.gamma', 0.1)


        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.loss = self.build_loss()

    
    def build_dataset(self, filename: str):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(self.dataset_root, filename), transforms)

    def build_model(self):
        """ 构建模型 """
        raise NotImplementedError

    def build_loss(self):
        """ 构建损失函数 """
        loss_cls = lazy_import_module('torch.nn', self.loss_select)
        loss_args = {'weight': torch.tensor(self.loss_weight, dtype=torch.float32)} if self.loss_weight else {}
        return loss_cls(**loss_args)

    def build_optimizer(self,model: torch.nn.Module):
        """ 构建优化器 """
        optimizer_cls = lazy_import_module('torch.optim', self.optimizer_select)
        return optimizer_cls(model.parameters(), lr=self.lr)

    def build_lr_scheduler(self,optimizer):
        """ 构建学习率调度器 """
        if self.lr_scheduler_select is None:
            return None
        scheduler_cls = lazy_import_module('torch.optim.lr_scheduler', self.lr_scheduler_select)
        return scheduler_cls(optimizer, step_size=self.step_size, gamma=self.gamma)
    
    def get_train_dataset(self):
        """ 获得训练集的方法 """
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.train_fpath)
        return self.train_dataset

    def get_dev_dataset(self):
        """ 获得验证集的方法 """
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.eval_fpath[0])
        return self.dev_dataset
    
    def get_test_dataset(self):
        """ 获得测试集的方法 """
        if self.test_dataset is None:
            self.test_dataset = self.build_dataset(self.eval_fpath[1])
        return self.test_dataset
    
    def get_batch_iterator(self, dataloader: DataLoader):
        """
        创建一个每次获取 batch_size 数据的迭代器。

        param dataloader (DataLoader): PyTorch 数据加载器，包含了训练数据。
        return:iterator: 每次返回 batch_size 大小数据的迭代器。
        TypeError: 如果 dataloader 不是 DataLoader 类型。
        """
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.DatasetLoader, but got {type(dataloader)}")
        
        return iter(dataloader)
        
    def build_sampler(dataset: Dataset, world_size: int, rank: int):
        """
        构建分布式数据加载器的采样器，根据给定的进程数量和进程编号，划分数据集并进行分布式训练。
        
        :param dataset: Dataset, 要进行分布式采样的数据集。
        :param world_size: int, 总进程数，即分布式训练的总工作节点数。
        :param rank: int, 当前进程的编号，用于标识不同的工作节点。
        :return: DistributedSampler, 用于分布式训练的数据采样器。
        :raises TypeError: 如果 dataset 不是 Dataset 类型。
        :raises ValueError: 如果 world_size 或 rank 的值不合法。
        :raises RuntimeError: 如果创建 DistributedSampler 失败。
        """

        # 检查dataset类型
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected 'dataset' to be of type Dataset, but got {type(dataset)}")

        # 检查world_size和rank是否为正整数
        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"Expected 'world_size' to be a positive integer, but got {world_size}")
        
        if not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise ValueError(f"Expected 'rank' to be a valid integer in the range [0, {world_size - 1}], but got {rank}")

        # 创建并返回DistributedSampler
        try:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        except Exception as e:
            raise RuntimeError(f"Failed to create DistributedSampler: {e}")

        return sampler

    def build_dataloader(dataset: Dataset, batch_size: int, sampler:Sampler):
        """
        创建数据加载器，用于根据给定的批大小和采样器加载数据。
        
        :param dataset: Dataset, 用于训练或评估的数据集。
        :param batch_size: int, 每个 batch 的数据量。
        :param sampler: Sampler, 用于数据分配的采样器。如果为 None，则默认按照顺序加载数据。
        :return: DataLoader, 用于加载数据的 DataLoader 实例。
        :raises TypeError: 如果 dataset 不是 Dataset 类型，或者 sampler 不是 Sampler 类型。
        :raises ValueError: 如果数据集为空并且没有提供采样器。
        :raises RuntimeError: 如果创建 DataLoader 失败。
        """
        # 检查dataset类型
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.Dataset, but got {type(dataset)}")
        
        # 检查batch_size类型
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Expected 'batch_size' to be a positive integer, but got {batch_size}")
        
        # 检查sampler类型
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"Expected 'sampler' to be of type torch.utils.data.Sampler, but got {type(sampler)}")
        
        # 如果sampler是None，确保dataset大小不为0
        if sampler is None and len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot create DataLoader without a sampler.")
        
        # 创建DataLoader
        # 如果sampler是None，保持数据原有顺序构造加载器
        if sampler is None:
            try:
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")

        else:
            try:
                data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")
        
        return data_loader
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

    
    def train_step(self, model: torch.nn.Module, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError