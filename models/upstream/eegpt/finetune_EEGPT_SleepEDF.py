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

from .Transformers.pos_embed import create_1d_absolute_sin_cos_embedding
from .models.EEGPT_mcae import EEGTransformer

from .Network.utils import Conv1dWithConstraint, LinearWithConstraint

use_channels_names = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ' ]

class LitEEGPTCausal(nn.Module):

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        self.num_class = 5
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, 256*30],
            patch_size=32*2,
            # patch_stride = 32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
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

        self.chan_conv       = Conv1dWithConstraint(2, self.chans_num, 1, max_norm=1)
        
        self.linear_probe1   = LinearWithConstraint(2048, 64, max_norm=1)
        self.drop            = torch.nn.Dropout(p=0.50)        
        self.decoder         = torch.nn.TransformerDecoder(
                                    decoder_layer=torch.nn.TransformerDecoderLayer(64, 4, 64*4, activation=torch.nn.functional.gelu, batch_first=False),
                                    num_layers=4
                                )
        self.cls_token =        torch.nn.Parameter(torch.rand(1,1,64)*0.001, requires_grad=True)
        self.linear_probe2   =   LinearWithConstraint(64, self.num_class, max_norm=0.25)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
    
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    def forward(self, x):
        B, C, T = x.shape
        x = temporal_interpolation(x, 256*30)
        x = self.chan_conv(x)
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        
        h = self.linear_probe1(self.drop(h))
        pos = create_1d_absolute_sin_cos_embedding(h.shape[1], dim=64)
        h = h + pos.repeat((h.shape[0], 1, 1)).to(h)
        
        h = torch.cat([self.cls_token.repeat((h.shape[0], 1, 1)).to(h.device), h], dim=1)
        h = h.transpose(0,1)
        h = self.decoder(h, h)[0,:,:]
        
        h = self.linear_probe2(h)
        return x, h

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
