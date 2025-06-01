#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : dalia_hr_task.py
@Time    : 2025/03/30 20:31:56
@Desc    : 
"""
import torch
import numpy as np
from tasks import PRLTask
from utils import get_nested_field, lazy_import_module


class DaLiaHREstimationTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(**self.model_params)
        return model
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        spec = sample['ppg_acc']
        times = sample['ppg_time']
        spec = spec.float()
        times = times.float()
        label = sample['hr']
        pred = model(spec, times)
        # print(label,pred)
        # print(label.shape, pred.shape)
        # print("Loss params:", self.loss.__dict__)
        loss = self.loss(pred, label)
        return{
            'loss': loss,
            # 'output': pred,
            'label': label
        }

    @torch.no_grad()
    def valid_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        spec = sample['ppg_acc']
        times = sample['ppg_time']
        spec = spec.float()
        times = times.float()
        label = sample['hr']
        # model.online(False)
        logits, (output, _) = model(spec, times, use_prior=True, need_logits_use_prior=True)
        loss = self.loss(logits, label)
        return{
            'loss': loss,
            'output': output,
            'label': label
        }
    
    def on_train_start(self, trainer, *args, **kwargs):
        with torch.no_grad():
            train_labels = []
            for sample in trainer.train_loader:
                train_labels.append(sample["hr"].numpy())
                # print(sample["hr"].shape)
        train_ys = np.concatenate(train_labels)
        trainer.model.fit_prior_layer([train_ys])