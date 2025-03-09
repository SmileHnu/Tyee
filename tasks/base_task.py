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
import numpy as np
from torch.utils.data import Dataset
from dataset.transform import Compose
from dataset.split import DatasetSplitter
from utils import lazy_import_module, get_nested_field
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler

class PRLTask(object):
    def __init__(self, cfg: dict) -> None:
        # 解析配置
        self.cfg = cfg
        
        # 数据集路径和变换
        self.root_path = get_nested_field(cfg, 'dataset.root_path', '')
        self.io_path = get_nested_field(cfg, 'dataset.io_path', '')
        self.io_mode = get_nested_field(cfg, 'dataset.io_mode', 'lmdb')

        self.dataset = get_nested_field(cfg, 'dataset.dataset', '')
        self.num_workers = get_nested_field(cfg, 'dataset.num_workers', 4)
        
        offline_transform_cfg = get_nested_field(cfg, 'dataset.offline_transform', None)
        self.offline_transform = self.build_transforms(offline_transform_cfg)
        online_transform_cfg = get_nested_field(cfg, 'dataset.online_transform', None)
        self.online_transform = self.build_transforms(online_transform_cfg)
        label_transform_cfg = get_nested_field(cfg, 'dataset.label_transform', None)
        self.label_transform = self.build_transforms(label_transform_cfg, is_label_transform=True)

        self.split_params = get_nested_field(cfg, 'dataset.split', {})

        # 模型配置
        self.downstream_classes = get_nested_field(cfg, 'model.downstream.classes', 1)
        self.downstream_select = get_nested_field(cfg, 'model.downstream.select', '')
        self.upstream_select = get_nested_field(cfg, 'model.upstream.select', '')
        self.upstream_trainable = get_nested_field(cfg, 'model.upstream.trainable', False)

        # 损失函数配置
        self.loss_select = get_nested_field(cfg, 'task.loss.select', '')
        self.loss_params = get_nested_field(cfg, 'task.loss', {})

        # 优化器配置
        self.optimizer_select = get_nested_field(cfg, 'optimizer.select', '')
        self.lr = get_nested_field(cfg, 'optimizer.lr', 0.0001)
        self.layer_decay = get_nested_field(cfg, 'optimizer.layer_decay', 1.0)
        self.weight_decay = get_nested_field(cfg, 'optimizer.weight_decay', 0.0)
        # 学习率调度器配置
        self.lr_scheduler_select = get_nested_field(cfg, 'lr_scheduler.select', '')
        self.lr_scheduler_params = get_nested_field(cfg, 'lr_scheduler', {})


        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.loss = self.build_loss()

    def build_transforms(self, transform_cfg: dict, is_label_transform: bool = False):
        """
        构建数据变换的方法。

        参数:
            transform_cfg (dict): 变换配置字典。
            is_label_transform (bool): 是否为标签变换。

        返回:
            dict: 包含 Compose 对象的字典。
        """
        transforms = {}
        if transform_cfg is not None:
            if is_label_transform:
                transform_objs = []
                for transform_name, params in transform_cfg.items():
                    transform_cls = lazy_import_module('dataset.transform', transform_name)
                    transform_objs.append(transform_cls(**params))
                return Compose(transform_objs)
            else:
                for key, transform_list in transform_cfg.items():
                    transform_objs = []
                    for transform_name, params in transform_list.items():
                        transform_cls = lazy_import_module('dataset.transform', transform_name)
                        transform_objs.append(transform_cls(**params))
                    transforms[key] = Compose(transform_objs)
                return transforms
        return None

    def build_dataset(self, root_path: str, io_path: str) -> Dataset:
        """
        构建数据集的方法。

        参数:
            root_path (str): 数据集的原数据路径。
            io_path (str): 数据集的统一数据存放路径。

        返回:
            Dataset: 构建好的数据集对象。
        """
        if root_path is None and io_path is None:
            return None

        Dataset = lazy_import_module('dataset', self.dataset)
        return Dataset(root_path=root_path, 
                       io_path=io_path, 
                       io_mode=self.io_mode, 
                       num_worker=self.num_workers, 
                       offline_transform=self.offline_transform,
                       online_transform=self.online_transform,
                       label_transform=self.label_transform)
    
    def build_datasets(self):
        """
        构建训练集、验证集和测试集的方法。

        返回:
            Tuple[Dataset, Dataset, Dataset]: 训练集、验证集和测试集。
        """
        train_dataset = self.build_dataset(self.root_path.get('train', None), self.io_path.get('train', None))
        dev_dataset = self.build_dataset(self.root_path.get('dev', None), self.io_path.get('dev', None))
        test_dataset = self.build_dataset(self.root_path.get('test', None), self.io_path.get('test', None))

        return train_dataset, dev_dataset, test_dataset

    def build_model(self):
        """ 构建模型 """
        raise NotImplementedError

    def build_loss(self):
        """ 构建损失函数 """
        try:
            loss_cls = lazy_import_module('torch.nn', self.loss_select)
        except (ImportError, AttributeError):
            loss_cls = lazy_import_module('criterions', self.loss_select)
        
        # 从配置中获取损失函数参数
        loss_params = {k: v for k, v in self.loss_params.items() if k not in ['select']}
        return loss_cls(**loss_params)

    def build_optimizer(self,model: torch.nn.Module):
        """ 构建优化器 """
        # 使用 set_optimizer_params 方法构造参数组
        param_groups = self.set_optimizer_params(model, self.lr, self.layer_decay, self.weight_decay)
        # 动态导入优化器类
        try:
            optimizer_cls = lazy_import_module('torch.optim', self.optimizer_select)
        except (ImportError, AttributeError):
            optimizer_cls = lazy_import_module('optim', self.optimizer_select)
    
        return optimizer_cls(param_groups)
    
    def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float = 1e-5):
        """
        根据 lr 和 layer_decay 设置模型每一层的学习率，构造对应的参数组
        :param model: 需要设置学习率的模型
        :param lr: 基础学习率
        :param layer_decay: 学习率衰减率
        :return: 参数组列表，用于 optimizer 的创建
        """
        param_groups = [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
        return param_groups

    def build_lr_scheduler(self,optimizer):
        """ 构建学习率调度器 """
        if self.lr_scheduler_select is None:
            return None
        try:
            scheduler_cls = lazy_import_module('torch.optim.lr_scheduler', self.lr_scheduler_select)
        except (ImportError, AttributeError):
            scheduler_cls = lazy_import_module('optim.lr_scheduler', self.lr_scheduler_select)
        
        # 从配置中获取调度器参数
        scheduler_params = {k: v for k, v in self.lr_scheduler_params.items() if k not in ['select']}
        return scheduler_cls(optimizer, **scheduler_params)
    
    def get_datasets(self,):
        """ 获取数据集 """
        self.train_dataset, self.dev_dataset, self.test_dataset = self.build_datasets()

        self.splitter = DatasetSplitter(train_dataset=self.train_dataset, dev_dataset=self.dev_dataset, test_dataset=self.test_dataset)

        splits = self.splitter.split(**self.split_params)

        return splits
    
    def collate_fn(self, batch):
        # 假设 batch 是一个包含字典的列表
        batch_signals = {key: [] for key in batch[0]}
        
        for item in batch:
            for key, value in item.items():
                batch_signals[key].append(value)
        
        # 将信号数据和标签分别拼接成一个批次
        for key in batch_signals:
            # print(key)
            # print(batch_signals[key])
            if isinstance(batch_signals[key][0], torch.Tensor):
                # print(1)
                batch_signals[key] = torch.stack(batch_signals[key])
            elif isinstance(batch_signals[key][0], np.ndarray):
                # print(2)
                batch_signals[key] = torch.tensor(np.stack(batch_signals[key]))
            elif isinstance(batch_signals[key][0], (int, float)) or isinstance(batch_signals[key][0], list):
                # print(3)
                # print(batch_signals[key].dtype)
                # 如果是数值列表，将其转换为张量
                batch_signals[key] = torch.tensor(batch_signals[key])
                # print(batch_signals[key].dtype)
            else:
                # print(4)
                batch_signals[key] = batch_signals[key]
        
        return batch_signals
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
        
    def build_sampler(self, dataset: Dataset, world_size: int, rank: int):
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

    def build_dataloader(self, dataset: Dataset, batch_size: int, sampler:Sampler, shuffle: bool = False):
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
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")

        else:
            try:
                data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=self.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")
        
        return data_loader
    
    def load_sample(self, iterator, device):
        sample = next(iterator)
            
        for k, v in sample.items():
            if v is not None:
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(device)
        
        return sample

    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        raise NotImplementedError
    
    def momentum_update(self, model: torch.nn.Module, *args, **kwargs):
        pass