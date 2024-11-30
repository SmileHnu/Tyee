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
from timm.models import create_model
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

class TUABTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.test_dataset = None
        self.dev_dataset = None
        self.model = get_nested_field(cfg, 'model.upstream.select', '')
        self.finetune = get_nested_field(cfg, 'model.upstream.finetune', '')
        self.nb_classes = get_nested_field(cfg, 'model.upstream.nb_classes', 0)
        self.drop = get_nested_field(cfg, 'model.upstream.drop', 0.0)
        self.drop_path = get_nested_field(cfg, 'model.upstream.drop_path', 0.0)
        self.attn_drop_rate = get_nested_field(cfg, 'model.upstream.attn_drop_rate', 0.0)
        self.drop_block_rate = get_nested_field(cfg, 'model.upstream.drop_block_rate', None)
        self.use_mean_pooling = get_nested_field(cfg, 'model.upstream.use_mean_pooling', False)
        self.init_scale = get_nested_field(cfg, 'model.upstream.init_scale', 0.001)
        self.rel_pos_bias = get_nested_field(cfg, 'model.upstream.rel_pos_bias', False)
        self.abs_pos_emb = get_nested_field(cfg, 'model.upstream.abs_pos_emb', False)
        self.layer_scale_init_value = get_nested_field(cfg, 'model.upstream.layer_scale_init_value', 0.0)
        self.qkv_bias = get_nested_field(cfg, 'model.upstream.qkv_bias', False)

        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

        self.input_chans = self.get_input_chans(ch_names)

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
        seed = 12345
        np.random.seed(seed)
        files = os.listdir(os.path.join(root, fpath))
        if fpath == "train":
            np.random.shuffle(files)
        Dataset = lazy_import_module('dataset', self.dataset)
        # transforms = [lazy_import_module('dataset.transforms', t) for t in self.transforms_select]
        return Dataset(os.path.join(root, fpath), files)

    def build_model(self):

        model = create_model(
        self.model,
        pretrained=True,
        checkpoint_path=self.finetune,
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
    
    def get_input_chans(ch_names):
        input_chans = [0] # for cls token
        for ch_name in ch_names:
            input_chans.append(standard_1020.index(ch_name) + 1)
        return input_chans