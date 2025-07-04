#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : tuev_task.py
@Time    : 2024/12/01 20:54:55
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

class TUEVTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None


        self.model_select = get_nested_field(cfg, 'model.select', '')
        self.finetune = get_nested_field(cfg, 'model.finetune', '')
        self.nb_classes = get_nested_field(cfg, 'model.nb_classes', 0)

        self.qkv_bias = get_nested_field(cfg, 'model.qkv_bias', False)
        self.rel_pos_bias = get_nested_field(cfg, 'model.rel_pos_bias', False)
        self.abs_pos_emb = get_nested_field(cfg, 'model.abs_pos_emb', True)
        self.layer_scale_init_value = get_nested_field(cfg, 'model.layer_scale_init_value', 0.1)

        self.input_size = get_nested_field(cfg, 'model.input_size', 200)
        self.drop = get_nested_field(cfg, 'model.drop', 0.0)
        self.attn_drop_rate = get_nested_field(cfg, 'model.attn_drop_rate', 0.0)
        self.drop_path = get_nested_field(cfg, 'model.drop_path', 0.1)

        self.disable_eval_during_finetuning = get_nested_field(cfg, 'model.disable_eval_during_finetuning', False)
        self.model_ema = get_nested_field(cfg, 'model.model_ema', False)
        self.model_ema_decay = get_nested_field(cfg, 'model.model_ema_decay', 0.9999)
        self.model_ema_force_cpu = get_nested_field(cfg, 'model.model_ema_force_cpu', False)


        self.finetune = get_nested_field(cfg, 'model.finetune', '')
        self.model_key = get_nested_field(cfg, 'model.model_key', 'model|module')
        self.model_prefix = get_nested_field(cfg, 'model.model_prefix', '')
        self.model_filter_name = get_nested_field(cfg, 'model.model_filter_name', 'gzp')
        self.init_scale = get_nested_field(cfg, 'model.init_scale', 0.001)
        self.use_mean_pooling = get_nested_field(cfg, 'model.use_mean_pooling', True)
        self.use_cls = get_nested_field(cfg, 'model.use_cls', False)
        self.disable_weight_decay_on_rel_pos_bias = get_nested_field(cfg, 'model.disable_weight_decay_on_rel_pos_bias', False)


        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        self.input_chans = self.get_input_chans(ch_names)

    def get_layer_id(self, var_name, num_max_layer):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return num_max_layer - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return num_max_layer - 1

    def set_optimizer_params(self, model: torch.nn.Module, lr: float, layer_decay: float, weight_decay: float):
        param_groups = []
        num_layers = model.get_num_layers()
        layer_scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        skip_list = ['pos_embed', 'cls_token', 'time_embed']
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  
            layer_id = self.get_layer_id(name, num_layers + 2)
            
            scale = layer_scales[layer_id]
            param_groups.append({
                'params': param, 
                # 'lr_scale': scale,
                'lr': lr * scale,
                'weight_decay': 0.0 if param.ndim <= 1 or name.endswith(".bias") or name in skip_list else weight_decay
            })
            print(f"Layer {layer_id}: {name}, lr_scale={scale}, lr={lr}, weight_decay={0.0 if param.ndim <= 1 or name.endswith('.bias') or name in skip_list else weight_decay}")
        return param_groups

    def build_model(self):
        
        module_name, class_name = self.model_select.rsplit('.', 1)
        model_name = lazy_import_module(f'models.{module_name}', class_name)
        model = model_name(
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

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        return model
        
    def train_step(self, model: nn.Module, sample: dict[str, torch.Tensor]):
        
        eeg = sample["eeg"]
        label = sample["event"]
        eeg = eeg.float() / 100
        eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg, self.input_chans)
        loss = self.loss(pred, label)
        return {
            "loss": loss,
            "output": pred,
            "label": label
        }

    @torch.no_grad()
    def valid_step(self, model, sample: dict[str, torch.Tensor]):
        eeg = sample["eeg"]
        label = sample["event"]
        eeg = eeg.float() / 100
        eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
        pred = model(eeg, self.input_chans)
        loss = self.loss(pred, label)

        return {
            "loss": loss,
            "output": pred,
            "label": label
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
