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
        # 加载权重
        checkpoint_path = "/home/lingyus/code/PRL/experiments/2025-05-17/23-26-38-dalia_hr_task.DaLiaHREstimationTask/checkpoint/fold_0/checkpoint_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
            print(f"Loaded model weights from {checkpoint_path}")
        else:
            print(f"No 'model' key found in checkpoint: {checkpoint_path}")
        # print(model)
        return model
    
    # def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float = 1e-5):
    #     """
    #     根据 lr 和 layer_decay 设置模型每一层的学习率，构造对应的参数组
    #     :param model: 需要设置学习率的模型
    #     :param lr: 基础学习率
    #     :param layer_decay: 学习率衰减率
    #     :return: 参数组列表，用于 optimizer 的创建
    #     """
    #     param_groups = [{'params': model.hybrid_unet.parameters(), 'lr': lr}]
    #     return param_groups
    
    def train_step(self, model: torch.nn.Module, sample: dict[str, torch.Tensor], *args, **kwargs):
        spec = sample['ppg_acc']
        times = sample['ppg_time']
        spec = spec.float()
        times = times.float()
        label = sample['hr'].float()
        pred = model(spec, times)
        # print(label,pred)
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
        label = sample['hr'].float()
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