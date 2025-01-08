#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : eegpt_tuev2_task.py
@Time    : 2024/12/30 20:07:27
@Desc    : 
"""

import os
import numpy as np
import torch
from tasks import PRLTask
from dataset import DatasetType
import torch.nn.functional as F
from utils import get_nested_field, lazy_import_module

class EEGPTTUEV2Task(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.checkpoint = get_nested_field(cfg, 'model.upstream.checkpoint', default=None)
        
    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.dataset_root, self.train_fpath)
        return self.train_dataset

    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.dataset_root, self.eval_fpath[0])
        return self.dev_dataset
    
    def get_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = self.build_dataset(self.dataset_root, self.eval_fpath[1])
        return self.test_dataset

    def build_dataset(self, root: str, fpath: str = "train"):
        """ 构建数据集 """
        seed = 4523
        np.random.seed(seed)
        files = os.listdir(os.path.join(root, fpath))
        Dataset = lazy_import_module('dataset', self.dataset)
        # transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(root, fpath), files)
    def build_model(self):
        model = lazy_import_module('models.upstream', self.upstream_select)
        return model(load_path = self.checkpoint)
    
    # def build_optimizer(self, model: torch.nn.Module):
    #     return model.optimizer
    def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float = 1e-5):
        """
        根据 lr 和 layer_decay 设置模型每一层的学习率，构造对应的参数组
        :param model: 需要设置学习率的模型
        :param lr: 基础学习率
        :param layer_decay: 学习率衰减率
        :return: 参数组列表，用于 optimizer 的创建
        """
        param_groups = []
        num_layers = model.get_num_layers()  # 获取模型的层数
        skip_list = ('target_encoder.summary_token', 'target_encoder.chan_embed')
        print(f"Total layers: {num_layers}")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结的参数
            # 获取层号
            layer_id = self.get_layer_id(name, num_layers)
            # 计算该层的学习率缩放比例
            scale = layer_decay ** (num_layers - layer_id - 1)
            # 构造参数组
            param_groups.append({
                'params': param, 
                'lr_scale': scale,
                'lr': lr,
                'weight_decay': 0.0 if param.ndim <= 1 or name.endswith(".bias") or name in skip_list else weight_decay
            })
            print(f"Layer {layer_id}: {name}, lr_scale={scale}, lr={lr}, weight_decay={0.0 if param.ndim <= 1 or name.endswith('.bias') or name in skip_list else weight_decay}")
        return param_groups
    def get_layer_id(self, var_name, num_max_layer):
        """
        获取参数所属的层号
        :param var_name: 参数名
        :param num_max_layer: 最大层数
        :return: 层号
        """
        if var_name.startswith("target_encoder.summary_token"):
            return 0
        elif var_name.startswith("target_encoder.patch_embed"):
            return 0
        elif var_name.startswith("target_encoder.chan_embed"):
            return 0
        if var_name.startswith("layer"):
            layer_id = int(var_name.split('.')[1])
            return layer_id
        elif "blocks" in var_name:
            # 提取 "blocks" 后面的部分
            parts = var_name.split('.')
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts):
                    layer_id = int(parts[i + 1])
                    return layer_id
        else:
            return num_max_layer - 1
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['x']
        target = sample['target']
        x = x.float() / 100
        x, pred = model(x)
        
        loss = self.loss(pred, target)
        
        return{
            'loss': loss,
            'output': pred,
            'target': target
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        x = sample['x']
        target = sample['target']
        x = x.float() / 100
        x, pred = model(x)
        
        loss = self.loss(pred, target)
        
        return{
            'loss': loss,
            'output': pred,
            'target': target
        }