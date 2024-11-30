#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : labram_task.py
@Time    : 2024/11/22 18:40:41
@Desc    : 
"""

import os
import torch
import numpy as np
from torch import nn
from pathlib import Path
from tasks import PRLTask
from timm.utils import ModelEma
from dataset import DatasetType
from timm.models import create_model
from collections import OrderedDict
from models.upstream import labram_base_patch200_200
from utils import lazy_import_module, get_nested_field



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

class MoBITask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None


        self.model = get_nested_field(cfg, 'model.upstream.select', '')
        self.finetune = get_nested_field(cfg, 'model.upstream.finetune', '')  # 获取微调的设置
        self.nb_classes = get_nested_field(cfg, 'model.upstream.nb_classes', 0)  # 获取类别数

        self.qkv_bias = get_nested_field(cfg, 'model.upstream.qkv_bias', True)  # 获取是否使用 qkv_bias
        self.rel_pos_bias = get_nested_field(cfg, 'model.upstream.rel_pos_bias', True)  # 获取是否使用相对位置偏置
        self.abs_pos_emb = get_nested_field(cfg, 'model.upstream.abs_pos_emb', False)  # 获取是否使用绝对位置嵌入
        self.layer_scale_init_value = get_nested_field(cfg, 'model.upstream.layer_scale_init_value', 0.1)  # 获取初始化的层比例值

        self.input_size = get_nested_field(cfg, 'model.upstream.input_size', 200)  # 获取输入大小（默认为200）
        self.drop = get_nested_field(cfg, 'model.upstream.drop', 0.0)  # 获取丢弃率
        self.attn_drop_rate = get_nested_field(cfg, 'model.upstream.attn_drop_rate', 0.0)  # 获取注意力丢弃率
        self.drop_path = get_nested_field(cfg, 'model.upstream.drop_path', 0.1)  # 获取drop path率

        self.disable_eval_during_finetuning = get_nested_field(cfg, 'model.upstream.disable_eval_during_finetuning', False)  # 是否禁用微调期间的评估

        self.model_ema = get_nested_field(cfg, 'model.upstream.model_ema', False)  # 是否使用模型EMA
        self.model_ema_decay = get_nested_field(cfg, 'model.upstream.model_ema_decay', 0.9999)  # EMA衰减系数
        self.model_ema_force_cpu = get_nested_field(cfg, 'model.upstream.model_ema_force_cpu', False)  # 是否强制使用CPU

        # 获取 finetuning 相关的配置参数，所有字段都放在 'model.upstream' 下
        self.finetune = get_nested_field(cfg, 'model.upstream.finetune', '')  # 获取微调的路径
        self.model_key = get_nested_field(cfg, 'model.upstream.model_key', 'model|module')  # 获取模型关键字
        self.model_prefix = get_nested_field(cfg, 'model.upstream.model_prefix', '')  # 获取模型前缀
        self.model_filter_name = get_nested_field(cfg, 'model.upstream.model_filter_name', 'gzp')  # 获取模型过滤名称
        self.init_scale = get_nested_field(cfg, 'model.upstream.init_scale', 0.001)  # 获取初始化的比例
        self.use_mean_pooling = get_nested_field(cfg, 'model.upstream.use_mean_pooling', True)  # 获取是否使用平均池化
        self.use_cls = get_nested_field(cfg, 'model.upstream.use_cls', False)  # 获取是否使用cls token
        self.disable_weight_decay_on_rel_pos_bias = get_nested_field(cfg, 'model.upstream.disable_weight_decay_on_rel_pos_bias', False)  # 获取是否禁用相对位置偏置的权重衰减


        # 60个EEG通道 ， 'PO3', 'POZ', 'PO4', 'PO8' EOG通道
        ch_names = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 
            'T7', 'C3', 'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 
            'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'AF7',
            'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 
            'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 
            'P5', 'P1', 'P2', 'P6', 'PO7'
        ]


        self.input_chans = self.get_input_chans(ch_names)

    def get_train_dataset(self):
        if self.train_dataset is None:
            self.train_dataset = self.build_dataset(self.dataset_root, DatasetType.TRAIN)
        return self.train_dataset

    def get_dev_dataset(self):
        if self.dev_dataset is None:
            self.dev_dataset = self.build_dataset(self.dataset_root, DatasetType.DEV)
        return self.dev_dataset
    
    def get_test_dataset(self):
        if self.test_dataset is None:
            self.test_dataset = self.build_dataset(self.dataset_root, DatasetType.TEST)
        return self.test_dataset

    def build_dataset(self, root: str, split: DatasetType = DatasetType.UNKNOWN):
        """ 构建数据集 """
        Dataset = lazy_import_module('dataset', self.dataset)
        # transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(root, split)

    def build_model(self):

        model = create_model(
        self.model,
        pretrained=False,
        num_classes=self.nb_classes,
        drop_rate=self.drop,
        drop_path_rate=self.drop_path,
        attn_drop_rate=self.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=self.use_mean_pooling,
        init_scale=self.init_scale,
        use_rel_pos_bias=self.rel_pos_bias,
        use_abs_pos_emb=self.abs_pos_emb,
        init_values=self.layer_scale_init_value,
        qkv_bias=self.qkv_bias,
        )

        patch_size = model.patch_size
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
            checkpoint_model = None
            for model_key in self.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None) and (self.model_filter_name != ''):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

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
        return model
        
    def train_step(self, model: nn.Module, sample: dict[str, torch.Tensor]):
        x = sample["x"]
        target = sample["target"]
        
        pred = model(x, self.input_chans)
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
        
        pred = model(x, self.input_chans)
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