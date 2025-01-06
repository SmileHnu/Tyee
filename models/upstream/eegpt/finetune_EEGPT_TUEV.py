import random 
import os
import torch
from torch import nn


from functools import partial
import numpy as np
import random
import os 
import tqdm

import torch.nn.functional as F

from .models.EEGPT_mcae import EEGTransformer

from .Network.utils import Conv1dWithConstraint, LinearWithConstraint

use_channels_names = [      
             'FP1','FPZ', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2' ]
        
ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

class LitEEGPTCausal(nn.Module):

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 1000],
            patch_size=32*2,
            embed_num=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        self.chan_conv       = Conv1dWithConstraint(len(ch_names), self.chans_num, 1, max_norm=1)
        self.linear_probe1   =   LinearWithConstraint(64, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(240, 6, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=True
        self.optimizer = torch.optim.AdamW(
            list(self.chan_conv.parameters())+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters())+
            list(self.target_encoder.parameters()),
            weight_decay=0.01)
        
    def forward(self, x):
        
        x = self.chan_conv(x)
        
        self.target_encoder.eval()
        
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        
        h = h.flatten(1)
        
        h = self.linear_probe2(h)
        
        return x, h
    def get_num_layers(self):
        """
        获取模型的总层数
        """
        return len(self.target_encoder.blocks) + 2