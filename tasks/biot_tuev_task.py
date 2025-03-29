#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : biot_tuev_task.py
@Time    : 2025/03/18 12:29:53
@Desc    : 
"""

import os
import torch
import math
import numpy as np
from torch import nn
from pathlib import Path
from tasks import PRLTask
from einops import rearrange
from collections import OrderedDict
from utils import lazy_import_module, get_nested_field

class Args:
    def __init__(self, cfg):
        # 优化器相关参数
        self.opt = get_nested_field(cfg, 'optimizer.opt', 'adamw')
        self.opt_eps = get_nested_field(cfg, 'optimizer.opt_eps', 1e-8)
        self.opt_betas = get_nested_field(cfg, 'optimizer.opt_betas', None)
        self.clip_grad = get_nested_field(cfg, 'optimizer.clip_grad', None)
        self.momentum = get_nested_field(cfg, 'optimizer.momentum', 0.9)
        self.weight_decay = get_nested_field(cfg, 'optimizer.weight_decay', 0.05)
        self.weight_decay_end = get_nested_field(cfg, 'optimizer.weight_decay_end', None)

        # 学习率相关参数
        self.lr = get_nested_field(cfg, 'optimizer.lr', 5e-4)
        self.layer_decay = get_nested_field(cfg, 'optimizer.layer_decay', 0.65)
        self.warmup_lr = get_nested_field(cfg, 'optimizer.warmup_lr', 1e-6)
        self.min_lr = get_nested_field(cfg, 'optimizer.min_lr', 1e-6)
        self.lr = float(self.lr)
        self.opt_eps = float(self.opt_eps)
        self.min_lr = float(self.min_lr)
        self.warmup_lr = float(self.warmup_lr)


class TUEVTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None


        self.model_select = get_nested_field(cfg, 'model.select', '')
        self.pretrain_model_path = get_nested_field(cfg, 'model.pretrain_model_path', '')  # 获取微调的设置
        self.n_classes = get_nested_field(cfg, 'model.n_classes', 0)  # 获取类别数
        self.in_channels = get_nested_field(cfg, 'model.in_channels', 18)
        self.token_size = get_nested_field(cfg, 'model.token_size', 32)
        self.hop_length = get_nested_field(cfg, 'model.hop_length', 16)
        self.sampling_rate = get_nested_field(cfg, 'model.sampling_rate', 200)

        self.args= Args(cfg)

    def build_model(self):
        
        model_name = lazy_import_module('models', self.model_select)
        model = model_name(
            n_classes=self.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=self.in_channels,
            n_fft=self.token_size,
            hop_length=self.hop_length,
        )
        if self.pretrain_model_path and (self.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(self.pretrain_model_path))
            print(f"load pretrain model from {self.pretrain_model_path}")

        
        return model
        
    def train_step(self, model: nn.Module, sample: dict[str, torch.Tensor]):
        
        eeg = sample["eeg"]['signals']
        label = sample["label"]
        eeg = eeg.float()
        # eeg = eeg.float() / 100
        # eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg)
        loss = self.loss(pred, label)
        return {
             "loss": loss,
            "output": pred,
            "label": label
        }

    @torch.no_grad()
    def valid_step(self, model, sample: dict[str, torch.Tensor]):
        eeg = sample["eeg"]['signals']
        label = sample["label"]
        eeg = eeg.float()
        # eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg)
        loss = self.loss(pred, label)

        return {
            "loss": loss,
            "output": pred,
            "label": label
        }
    
