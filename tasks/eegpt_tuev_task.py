#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : eegpt_tuev_task.py
@Time    : 2024/12/29 12:41:46
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
from timm.utils import ModelEma
from dataset import DatasetType
from timm.models import create_model
from collections import OrderedDict
from models.upstream import labram_base_patch200_200
from utils import lazy_import_module, get_nested_field
from timm.loss import LabelSmoothingCrossEntropy
from models.upstream.labram.optim_factory import create_optimizer,LayerDecayValueAssigner

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

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


class EEGPTTUEVTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None


        self.model = get_nested_field(cfg, 'model.upstream.select', '')
        self.finetune = get_nested_field(cfg, 'model.upstream.finetune', '')  # 获取微调的设置
        self.nb_classes = get_nested_field(cfg, 'model.upstream.nb_classes', 0)  # 获取类别数

        self.qkv_bias = get_nested_field(cfg, 'model.upstream.qkv_bias', False)  # 获取是否使用 qkv_bias
        self.rel_pos_bias = get_nested_field(cfg, 'model.upstream.rel_pos_bias', False)  # 获取是否使用相对位置偏置
        self.abs_pos_emb = get_nested_field(cfg, 'model.upstream.abs_pos_emb', True)  # 获取是否使用绝对位置嵌入
        self.layer_scale_init_value = get_nested_field(cfg, 'model.upstream.layer_scale_init_value', 0.1)  # 获取初始化的层比例值

        self.input_size = get_nested_field(cfg, 'model.upstream.input_size', 200)  # 获取输入大小（默认为200）
        self.drop = get_nested_field(cfg, 'model.upstream.drop', 0.0)  # 获取丢弃率
        self.attn_drop_rate = get_nested_field(cfg, 'model.upstream.attn_drop_rate', 0.0)  # 获取注意力丢弃率
        self.drop_path = get_nested_field(cfg, 'model.upstream.drop_path', 0.1)  # 获取drop path率

        self.disable_eval_during_finetuning = get_nested_field(cfg, 'model.upstream.disable_eval_during_finetuning', False)  # 是否禁用微调期间的评估

        self.model_ema = get_nested_field(cfg, 'model.upstream.model_ema', False)  # 是否使用模型EMA
        self.model_ema_decay = get_nested_field(cfg, 'model.upstream.model_ema_decay', 0.9999)  # EMA衰减系数
        self.model_ema_force_cpu = get_nested_field(cfg, 'model.upstream.model_ema_force_cpu', False)  # 是否强制使用CPU

        self.args= Args(cfg)

        # 获取 finetuning 相关的配置参数，所有字段都放在 'model.upstream' 下
        self.finetune = get_nested_field(cfg, 'model.upstream.finetune', '')  # 获取微调的路径
        self.model_key = get_nested_field(cfg, 'model.upstream.model_key', 'model|module')  # 获取模型关键字
        self.model_prefix = get_nested_field(cfg, 'model.upstream.model_prefix', '')  # 获取模型前缀
        self.model_filter_name = get_nested_field(cfg, 'model.upstream.model_filter_name', 'gzp')  # 获取模型过滤名称
        self.init_scale = get_nested_field(cfg, 'model.upstream.init_scale', 0.001)  # 获取初始化的比例
        self.use_mean_pooling = get_nested_field(cfg, 'model.upstream.use_mean_pooling', True)  # 获取是否使用平均池化
        self.use_cls = get_nested_field(cfg, 'model.upstream.use_cls', False)  # 获取是否使用cls token
        self.disable_weight_decay_on_rel_pos_bias = get_nested_field(cfg, 'model.upstream.disable_weight_decay_on_rel_pos_bias', False)  # 获取是否禁用相对位置偏置的权重衰减


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

    def build_lr_scheduler(self, optimizer):
        lr_scheduler_select = get_nested_field(self.cfg, 'lr_scheduler.select', '')
        if lr_scheduler_select == 'WarmupCosineAnnealingLR':
            T_max = get_nested_field(self.cfg, 'lr_scheduler.T_max', 100)
            eta_min = get_nested_field(self.cfg, 'lr_scheduler.eta_min', 0)
            warmup_start_lr = get_nested_field(self.cfg, 'lr_scheduler.warmup_start_lr', 0)
            warmup_steps = get_nested_field(self.cfg, 'lr_scheduler.warmup_steps', 0)
            return WarmupCosineAnnealingLR(optimizer=optimizer, T_max=T_max, warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, eta_min=eta_min)

    def build_optimizer(self, model):
        return self.optimizer

    def build_model(self):

        model_name = lazy_import_module('models.upstream', self.upstream_select)
        use_channels_names = [      
             'FP1','FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]
        
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                        'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        model =  model_name(
                num_classes=self.nb_classes,
                in_channels=len(ch_names), 
                img_size=[len(use_channels_names),1000], 
                use_channels_names=use_channels_names, 
                use_chan_conv=True,
                use_mean_pooling=self.use_mean_pooling,)

        patch_size = 200
        print("Patch size = %s" % str(patch_size))
        self.window_size = (1, self.input_size // patch_size)
        self.patch_size = patch_size

        if self.finetune:
            if self.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.finetune, map_location='cpu')

            print("Load ckpt from %s" % self.finetune)
            
            checkpoint_model = checkpoint['state_dict']
            self.load_state_dict(model, checkpoint_model, prefix=self.model_prefix)

        model_ema = None
        if self.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=self.model_ema_decay,
                device='cpu' if self.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % self.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        num_layers = model_without_ddp.get_num_layers()
        if self.args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(self.args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        if self.disable_weight_decay_on_rel_pos_bias:
            for i in range(num_layers):
                skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)
        self.optimizer = create_optimizer(
            self.args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        
        return model
        
    def train_step(self, model: nn.Module, sample: dict[str, torch.Tensor]):
        x = sample["x"]
        target = sample["target"]
        x = x.float() / 1000
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        pred = model(x)
        loss = self.loss(pred, target)
        return {
             "loss": loss,
            "output": pred,
            "target": target
        }

    @torch.no_grad()
    def valid_step(self, model, sample: dict[str, torch.Tensor]):
        x = sample["x"]
        target = sample["target"]
        x = x.float() / 1000
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        # 将输入数据转换为 Half 类型
        if x.dtype != torch.float16:
            x = x.half()
        pred = model(x)
        loss = self.loss(pred, target)

        return {
            "loss": loss,
            "output": pred,
            "target": target
        }
    
    def get_input_chans(self, ch_names):
        input_chans = [0] # for cls token
        for ch_name in ch_names:
            input_chans.append(standard_1020.index(ch_name) + 1)
        return input_chans
    
    def load_state_dict(self, model, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, warmup_steps, warmup_start_lr=0, eta_min=0, last_epoch=-1):
        """
        初始化 Warmup 和 CosineAnnealing 的学习率调度器，使用 lr_scale 而不是 layer_decay 来调整学习率。

        :param optimizer: 优化器
        :param T_max: CosineAnnealing 的最大步数
        :param warmup_steps: 预热阶段的步数
        :param warmup_start_lr: 预热阶段的初始学习率
        :param eta_min: 余弦退火阶段的最小学习率
        :param last_epoch: 上一次更新的步数，默认为-1
        """
        self.warmup_steps = warmup_steps  # 预热的步数
        self.T_max = T_max  # CosineAnnealing 的总步数
        self.eta_min = eta_min  # 最小学习率
        self.warmup_start_lr = warmup_start_lr  # 预热的起始学习率
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 获取当前步数
        step = self.last_epoch
        base_lr = self.base_lrs[0]  # 假设所有层使用相同的基本学习率

        # 线性 warmup 阶段
        if step < self.warmup_steps:
            lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * step / self.warmup_steps
        else:
            # CosineAnnealing 阶段
            step_after_warmup = step - self.warmup_steps
            lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * step_after_warmup / (self.T_max - self.warmup_steps)))

        # 获取每层的 lr_scale
        layer_lr_list = []
        for param_group in self.optimizer.param_groups:
            
            # 计算该层的学习率，使用 lr_scale 来调整
            lr_scale = param_group.get('lr_scale', 1.0)  # 默认 lr_scale 为 1.0
            adjusted_lr = lr * lr_scale  # 使用 lr_scale 调整学习率
            # 更新该层的学习率
            layer_lr_list.append(adjusted_lr)

        # 返回每个参数组的最终学习率
        return layer_lr_list