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
import multiprocessing
from torch.utils.data import Dataset
from tyee.utils import lazy_import_module, get_nested_field
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler
from typing import Tuple, List, Dict, Any

import logging
log = logging.getLogger(__name__)

class BaseTask(object):
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
        
        # Pre-parse input/target maps once during initialization
        self.input_map = get_nested_field(self.cfg, 'task.model.input_map', None)
        self.target_map = get_nested_field(self.cfg, 'task.target_map', None)

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
        """
        Build model dynamically based on config.
        This generic implementation should work for most cases.
        """
        if not self.model_select:
            raise ValueError("Model selection ('model.select') not specified in config.")
        
        try:
            module_name, class_name = self.model_select.rsplit('.', 1)
            model_cls = lazy_import_module(f'models.{module_name}', class_name)
        except (ValueError, ImportError) as e:
            raise ImportError(f"Failed to import model '{self.model_select}'. Error: {e}")

        return model_cls(**self.model_params)

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
        Set learning rate for each layer of the model based on lr and layer_decay, construct corresponding parameter groups.
        Supports:
        1. Layer-wise learning rate decay (if layer_decay < 1.0)
        2. Parameter freezing (via config 'optimizer.freeze_patterns')
        3. Custom weight decay rules (e.g. skip bias/norm)
        """
        import re
        
        # 1. Get freeze patterns from config
        freeze_patterns = get_nested_field(self.cfg, 'optimizer.freeze_patterns', [])
        if isinstance(freeze_patterns, str):
            freeze_patterns = [freeze_patterns]
            
        # 2. Freeze parameters
        if freeze_patterns:
            for name, param in model.named_parameters():
                for pattern in freeze_patterns:
                    if re.search(pattern, name):
                        param.requires_grad = False
                        log.info(f"Freezing parameter: {name}")
                        break

        # 3. Layer-wise decay logic
        # If layer_decay is 1.0 (default), we use simple grouping
        if layer_decay == 1.0:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay}]
            return param_groups

        # If layer_decay < 1.0, we need model-specific layer ID logic
        # We try to call model.get_layer_id(name, num_layers) if it exists
        # Otherwise we fall back to a simple heuristic or error
        
        if not hasattr(model, 'get_num_layers'):
             log.warning("Model does not have 'get_num_layers' method, but layer_decay < 1.0. Falling back to global LR.")
             return [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay}]

        if not hasattr(model, 'get_layer_id') and not hasattr(self, 'get_layer_id'):
             log.warning("Model/Task does not have 'get_layer_id' method, but layer_decay < 1.0. Falling back to global LR.")
             return [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay}]

        num_layers = model.get_num_layers()
        layer_scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        
        # Common skip list for weight decay
        skip_list = set()
        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
        
        param_groups = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer ID
            if hasattr(model, 'get_layer_id'):
                layer_id = model.get_layer_id(name, num_layers + 2)
            else:
                layer_id = self.get_layer_id(name, num_layers + 2)
            
            scale = layer_scales[layer_id]
            
            # Determine weight decay
            # Skip weight decay for 1D params (bias, norm) and special tokens
            if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
                this_wd = 0.0
            else:
                this_wd = weight_decay
                
            param_groups.append({
                'params': param,
                'lr': lr * scale,
                'weight_decay': this_wd
            })
            # log.debug(f"Layer {layer_id}: {name}, lr_scale={scale:.4f}, lr={lr*scale:.2e}, wd={this_wd}")
            
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
        if self.num_workers == 0:
            log.warning("num_workers is set to 0. This might become a bottleneck. "
                        "Consider setting a positive value for num_workers in your config file for better performance.")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.Dataset, but got {type(dataset)}")
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Expected 'batch_size' to be a positive integer, but got {batch_size}")
        
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"Expected 'sampler' to be of type torch.utils.data.Sampler, but got {type(sampler)}")
        
        if sampler is None and len(dataset) == 0:
            raise ValueError("Dataset is empty, cannot create DataLoader without a sampler.")
        
        # Determine multiprocessing context
        mp_context = None
        if self.num_workers > 0 and multiprocessing.get_start_method(allow_none=True) != 'fork':
            # If we are in a spawned process (like DDP), try to use fork for workers if possible
            # This avoids re-pickling the entire dataset for each worker
            try:
                mp_context = multiprocessing.get_context('fork')
            except ValueError:
                pass

        if sampler is None:
            try:
                data_loader = DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         shuffle=shuffle, 
                                         num_workers=self.num_workers,
                                         pin_memory=True,
                                         persistent_workers=True if self.num_workers > 0 else False,
                                         collate_fn=dataset.collate_fn,
                                         multiprocessing_context=mp_context)
            except Exception as e:
                raise RuntimeError(f"Failed to create DataLoader: {e}")

        else:
            try:
                data_loader = DataLoader(dataset, 
                                         batch_size=batch_size, 
                                         sampler=sampler,
                                         num_workers=self.num_workers,
                                         pin_memory=True,
                                         persistent_workers=True if self.num_workers > 0 else False,
                                         collate_fn=dataset.collate_fn,
                                         multiprocessing_context=mp_context)
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

    
    def _map_inputs(self, sample, mapping):
        """
        Helper to map sample data to function arguments based on configuration.
        mapping can be:
        - list: ['ecg', 'mask'] -> returns args=[sample['ecg'], sample['mask']], kwargs={}
        - dict: {'x': 'ecg', 'mask': 'mask'} -> returns args=[], kwargs={'x': sample['ecg'], ...}
        """
        if isinstance(mapping, list):
            args = [sample[k] for k in mapping]
            return args, {}
        elif isinstance(mapping, dict):
            kwargs = {k: sample[v] for k, v in mapping.items()}
            return [], kwargs
        return [], {}

    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        """
        Generic train_step that uses config to map inputs.
        """
        # 1. Prepare Model Inputs
        if self.input_map is None:
            # Fallback for legacy subclasses that might rely on overriding this method
            # If they didn't override and didn't provide config, we can't proceed.
            raise NotImplementedError("train_step is not implemented and 'task.model.input_map' is not configured.")
        
        model_args, model_kwargs = self._map_inputs(sample, self.input_map)
        
        # 2. Forward Pass
        # Ensure inputs are float if they are tensors (common requirement)
        # You can also use transforms for this, but a safety check here is helpful
        model_args = [arg.float() if isinstance(arg, torch.Tensor) and arg.dtype != torch.float32 else arg for arg in model_args]
        model_kwargs = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype != torch.float32 else v) for k, v in model_kwargs.items()}
        
        pred = model(*model_args, **model_kwargs)

        # 3. Prepare Loss Inputs
        # Default: assume loss takes (pred, target)
        
        if self.target_map:
            target_args, target_kwargs = self._map_inputs(sample, self.target_map)
            loss = self.loss(pred, *target_args, **target_kwargs)
            # Assuming the first target is the main label for metrics/logging
            label = target_args[0] if target_args else list(target_kwargs.values())[0]
        else:
            # Naive fallback: try to find a 'label' or 'target' key
            if 'label' in sample:
                label = sample['label']
            elif 'target' in sample:
                label = sample['target']
            else:
                # If we can't find a label, maybe the loss doesn't need one (unsupervised)?
                # But usually it does. Let's assume None and let Loss handle it or crash.
                label = None
            
            if label is not None:
                loss = self.loss(pred, label)
            else:
                loss = self.loss(pred)

        return {
            'loss': loss,
            'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        """
        Generic valid_step that uses config to map inputs.
        """
        # Logic is identical to train_step, but wrapped in no_grad (handled by decorator)
        # We duplicate logic to allow independent customization if needed, 
        # but for the generic case, it's the same mapping.
        
        if self.input_map is None:
             raise NotImplementedError("valid_step is not implemented and 'task.model.input_map' is not configured.")

        model_args, model_kwargs = self._map_inputs(sample, self.input_map)
        
        model_args = [arg.float() if isinstance(arg, torch.Tensor) and arg.dtype != torch.float32 else arg for arg in model_args]
        model_kwargs = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype != torch.float32 else v) for k, v in model_kwargs.items()}

        pred = model(*model_args, **model_kwargs)

        if self.target_map:
            target_args, target_kwargs = self._map_inputs(sample, self.target_map)
            loss = self.loss(pred, *target_args, **target_kwargs)
            label = target_args[0] if target_args else list(target_kwargs.values())[0]
        else:
            if 'label' in sample:
                label = sample['label']
            elif 'target' in sample:
                label = sample['target']
            else:
                label = None
            
            if label is not None:
                loss = self.loss(pred, label)
            else:
                loss = self.loss(pred)

        return {
            'loss': loss,
            'output': pred,
            'label': label
        }
    
    def on_train_start(self, trainer, *args, **kwargs):
        """Processing before each training process starts"""
        # Unwrap DDP if necessary to get the actual model
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        
        # Delegate to model if it has a hook
        if hasattr(model, 'on_train_start'):
            log.info(f"Invoking {model.__class__.__name__}.on_train_start hook")
            model.on_train_start(trainer.train_loader)

    def on_train_end(self, trainer, *args, **kwargs):
        """Processing after each training process ends"""
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if hasattr(model, 'on_train_end'):
            model.on_train_end()

    def on_train_epoch_start(self, trainer, step: int, *args, **kwargs):
        """Processing before each training epoch starts"""
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if hasattr(model, 'on_train_epoch_start'):
            model.on_train_epoch_start(step)

    def on_train_epoch_end(self, trainer, step: int, *args, **kwargs):
        """Processing after each training epoch ends"""
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if hasattr(model, 'on_train_epoch_end'):
            model.on_train_epoch_end(step)

    def on_valid_start(self, trainer, *args, **kwargs):
        """Processing before each validation process starts"""
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if hasattr(model, 'on_valid_start'):
            model.on_valid_start()

    def on_valid_end(self, trainer, *args, **kwargs):
        """Processing after each validation process ends"""
        model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        if hasattr(model, 'on_valid_end'):
            model.on_valid_end()