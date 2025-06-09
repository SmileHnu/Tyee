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
from utils import lazy_import_module, get_nested_field
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler
from typing import Tuple, List, Dict, Any

class PRLTask(object):
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
        self.root_path = get_nested_field(cfg, 'dataset.root_path', '')
        self.io_path = get_nested_field(cfg, 'dataset.io_path', '')
        self.num_workers = get_nested_field(cfg, 'dataset.num_workers', 0)
        self.dataset = get_nested_field(cfg, 'dataset.dataset', '')
        
        before_segment_transform_cfg = get_nested_field(cfg, 'dataset.before_segment_transform', None)
        self.before_segment_transform = self.build_transforms(before_segment_transform_cfg)
        offline_signal_transform_cfg = get_nested_field(cfg, 'dataset.offline_signal_transform', None)
        self.offline_signal_transform = self.build_transforms(offline_signal_transform_cfg)
        offline_label_transform_cfg = get_nested_field(cfg, 'dataset.offline_label_transform', None)
        self.offline_label_transform = self.build_transforms(offline_label_transform_cfg)
        online_signal_transform_cfg = get_nested_field(cfg, 'dataset.online_signal_transform', None)
        self.online_signal_transform = self.build_transforms(online_signal_transform_cfg)
        online_label_transform_cfg = get_nested_field(cfg, 'dataset.online_label_transform', None)
        self.online_label_transform = self.build_transforms(online_label_transform_cfg)

        self.dataset_cfg = get_nested_field(cfg, 'dataset', {})
        self.known_fields = {
            'batch_size', 'dataset', 'num_workers', 'root_path', 'io_path', 
            'split', 'before_segment_transform', 'offline_signal_transform', 
            'offline_label_transform', 'online_signal_transform', 'online_label_transform'
        }
        self.dataset_params = {k: v for k, v in self.dataset_cfg.items() 
                            if k not in self.known_fields}

        self.split_select = get_nested_field(cfg, 'dataset.split.select', 'NoSplit')
        self.split_init_params = get_nested_field(cfg, 'dataset.split.init_params', {})
        self.split_run_params = get_nested_field(cfg, 'dataset.split.run_params', {})

        self.model_select = get_nested_field(cfg, 'model.select', '')
        model_params = get_nested_field(cfg, 'model', {})
        self.model_params = {k: v for k, v in model_params.items() if k not in ['select']}

        self.loss_select = get_nested_field(cfg, 'task.loss.select', '')
        self.loss_params = get_nested_field(cfg, 'task.loss', {})

        self.optimizer_select = get_nested_field(cfg, 'optimizer.select', None)
        self.lr = get_nested_field(cfg, 'optimizer.lr', 0.0001)
        self.layer_decay = get_nested_field(cfg, 'optimizer.layer_decay', 1.0)
        self.weight_decay = get_nested_field(cfg, 'optimizer.weight_decay', 0.0)

        self.lr_scheduler_select = get_nested_field(cfg, 'lr_scheduler.select', None)
        self.lr_scheduler_params = get_nested_field(cfg, 'lr_scheduler', {})


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.loss = self.build_loss()

    def build_transforms(self, transform_cfg: list):
        """
        Method to build data transforms.

        Args:
            transform_cfg (list): List of transform configurations.

        Returns:
            Compose: Compose object or None.
        """
        if not transform_cfg:
            return None

        def instantiate_transform(cfg):
            cls = lazy_import_module('dataset.transform', cfg['select'])
            params = {k: v for k, v in cfg.items() if k not in ['select', 'transforms']}
            if 'transforms' in cfg and isinstance(cfg['transforms'], list):
                params['transforms'] = [instantiate_transform(sub_cfg) for sub_cfg in cfg['transforms']]
            return cls(**params)
        return [instantiate_transform(item) for item in transform_cfg]

    def build_dataset(self, root_path: str, io_path: str) -> Dataset:
        """
        Method to build dataset.

        Args:
            root_path (str): Original data path of the dataset.
            io_path (str): Unified data storage path of the dataset.

        Returns:
            Dataset: Built dataset object.
        """
        print("build_dataset root_path:", root_path)

        if root_path is None and io_path is None:
            return None

        module_name, class_name = self.dataset.rsplit('.', 1)
        Dataset = lazy_import_module(f'dataset.{module_name}', class_name)
        return Dataset(root_path=root_path, 
                       io_path=io_path, 
                       num_worker=self.num_workers,
                       before_segment_transform=self.before_segment_transform,
                       offline_signal_transform=self.offline_signal_transform,
                       offline_label_transform=self.offline_label_transform,
                       online_signal_transform=self.online_signal_transform,
                       online_label_transform=self.online_label_transform,
                       **self.dataset_params)
    
    def build_datasets(self):
        """
        Method to build training, validation and test datasets.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Training, validation and test datasets.
        """
        train_dataset = self.build_dataset(self.root_path.get('train', None), self.io_path.get('train', None))
        val_dataset = self.build_dataset(self.root_path.get('val', None), self.io_path.get('val', None))
        test_dataset = self.build_dataset(self.root_path.get('test', None), self.io_path.get('test', None))

        return train_dataset, val_dataset, test_dataset
    
    def build_splitter(self):
        """
        Method to build dataset splitter.

        Returns:
            DatasetSplitter: Dataset splitter object.
        """
        splitter_cls = lazy_import_module('dataset.split', self.split_select)
        
        return splitter_cls(**self.split_init_params)

    def build_model(self):
        """Build model"""
        raise NotImplementedError

    def build_loss(self):
        """Build loss function"""
        try:
            loss_cls = lazy_import_module('torch.nn', self.loss_select)
        except (ImportError, AttributeError):
            loss_cls = lazy_import_module('criterions', self.loss_select)
        
        loss_params = {k: v for k, v in self.loss_params.items() if k not in ['select']}
        if 'weight' in loss_params and loss_params['weight'] is not None:
            if isinstance(loss_params['weight'], (list, tuple)):
                loss_params['weight'] = torch.tensor(loss_params['weight'], dtype=torch.float32)
        return loss_cls(**loss_params)

    def build_optimizer(self,model: torch.nn.Module):
        """Build optimizer"""
        param_groups = self.set_optimizer_params(model, self.lr, self.layer_decay, self.weight_decay)
        try:
            optimizer_cls = lazy_import_module('torch.optim', self.optimizer_select)
        except (ImportError, AttributeError):
            optimizer_cls = lazy_import_module('optim', self.optimizer_select)
    
        return optimizer_cls(param_groups)
    
    def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float = 1e-5):
        """
        Set learning rate for each layer of the model based on lr and layer_decay, construct corresponding parameter groups
        :param model: Model that needs learning rate setting
        :param lr: Base learning rate
        :param layer_decay: Learning rate decay rate
        :return: List of parameter groups for optimizer creation
        """
        param_groups = [{'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
        return param_groups

    def build_lr_scheduler(self, optimizer, niter_per_epoch):
        """Build learning rate scheduler"""
        if self.lr_scheduler_select is None:
            return None
        
        scheduler_cls = lazy_import_module('optim.lr_scheduler', self.lr_scheduler_select)
        
        scheduler_params = {k: v for k, v in self.lr_scheduler_params.items() if k not in ['select']}
        return scheduler_cls(optimizer, niter_per_epoch, **scheduler_params)
    
    def get_datasets(self,):
        """Get datasets"""
        self.train_dataset, self.val_dataset, self.test_dataset = self.build_datasets()
        splitter = self.build_splitter()
        splits = list(splitter.split(
                            self.train_dataset, 
                            self.val_dataset, 
                            self.test_dataset, 
                            **self.split_run_params
        ))
        print("splits:", len(splits))
        return splits
          
    def get_batch_iterator(self, dataloader: DataLoader):
        """
        Create an iterator that fetches batch_size data each time.

        param dataloader (DataLoader): PyTorch data loader containing training data.
        return:iterator: Iterator that returns batch_size data each time.
        TypeError: If dataloader is not of DataLoader type.
        """
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.DatasetLoader, but got {type(dataloader)}")
        
        return iter(dataloader)
        
    def build_sampler(self, dataset: Dataset, world_size: int, rank: int):
        """
        Build sampler for distributed data loader, partition dataset and conduct distributed training based on given process count and process ID.
        
        :param dataset: Dataset, dataset for distributed sampling.
        :param world_size: int, total process count, i.e., total worker nodes for distributed training.
        :param rank: int, current process ID, used to identify different worker nodes.
        :return: DistributedSampler, data sampler for distributed training.
        :raises TypeError: If dataset is not of Dataset type.
        :raises ValueError: If world_size or rank values are invalid.
        :raises RuntimeError: If creating DistributedSampler fails.
        """
        if world_size == 1:
            return None
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected 'dataset' to be of type Dataset, but got {type(dataset)}")

        if not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"Expected 'world_size' to be a positive integer, but got {world_size}")
        
        if not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise ValueError(f"Expected 'rank' to be a valid integer in the range [0, {world_size - 1}], but got {rank}")

        try:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        except Exception as e:
            raise RuntimeError(f"Failed to create DistributedSampler: {e}")

        return sampler

    def build_dataloader(self, dataset: Dataset, batch_size: int, sampler:Sampler, shuffle: bool = False):
        """
        Create data loader for loading data based on given batch size and sampler.
        
        :param dataset: Dataset, dataset for training or evaluation.
        :param batch_size: int, amount of data per batch.
        :param sampler: Sampler, sampler for data allocation. If None, data is loaded in order by default.
        :return: DataLoader, DataLoader instance for loading data.
        :raises TypeError: If dataset is not of Dataset type, or sampler is not of Sampler type.
        :raises ValueError: If dataset is empty and no sampler is provided.
        :raises RuntimeError: If creating DataLoader fails.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.Dataset, but got {type(dataset)}")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Expected 'batch_size' to be a positive integer, but got {batch_size}")
        
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"Expected 'sampler' to be of type torch.utils.data.Sampler, but got {type(sampler)}")
        
        if sampler is None and len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot create DataLoader without a sampler.")
        
        if sampler is None:
            try:
                data_loader = DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=shuffle, 
                                         num_workers=self.num_workers, 
                                         collate_fn=dataset.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")

        else:
            try:
                data_loader = DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         sampler=sampler,
                                         num_workers=self.num_workers, 
                                         collate_fn=dataset.collate_fn)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")
        
        return data_loader
    
    def load_sample(self, iterator, device):
        sample = next(iterator)
        
        def recursive_to_device(data, device):
            if isinstance(data, dict):
                return {k: recursive_to_device(v, device) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_to_device(v, device) for v in data]
            elif isinstance(data, torch.Tensor):
                return data.to(device)
            else:
                return data
        
        sample = recursive_to_device(sample, device)
        
        return sample

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        raise NotImplementedError
    
    def on_train_start(self, trainer, *args, **kwargs):
        """Processing before each training process starts"""
        pass

    def on_train_end(self, trainer, *args, **kwargs):
        """Processing after each training process ends"""
        pass

    def on_train_epoch_start(self, trainer, step: int, *args, **kwargs):
        """Processing before each training epoch starts"""
        pass

    def on_train_epoch_end(self, trainer, step: int, *args, **kwargs):
        """Processing after each training epoch ends"""
        pass

    def on_valid_start(self, trainer, *args, **kwargs):
        """Processing before each validation process starts"""
        pass

    def on_valid_end(self, trainer, *args, **kwargs):
        """Processing after each validation process ends"""
        pass