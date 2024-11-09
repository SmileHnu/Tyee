#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : decoder.py
@Time    : 2024/09/11 21:04:51
@Desc    : 
"""
from torch import nn
from .utils import SamePad, Transpose


class Decoder1d(nn.Module):
    def __init__(
        self,
        decoder_dim: int = 384,
        decoder_groups: int = 16,
        decoder_kernel: int = 5,
        decoder_layers: int = 5,
        decoder_residual: bool = True,
        input_drop_prob: float = 0.1,
        proj_layers: int = 1,
        proj_ratio: float = 2.0,
        position_mask: bool = False,
        position_all: bool = False,
        input_dim: int = 768
    ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.decoder_groups = decoder_groups
        self.decoder_kernel = decoder_kernel
        self.decoder_layers = decoder_layers
        self.decoder_residual = decoder_residual
        self.input_drop_prob = input_drop_prob
        self.proj_layers = proj_layers
        self.proj_ratio = proj_ratio
        self.position_mask = position_mask
        self.position_all = position_all
        self.input_dim = input_dim

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    decoder_dim,
                    kernel_size=decoder_kernel,
                    padding=decoder_kernel // 2,
                    groups=decoder_groups,
                ),
                SamePad(decoder_kernel),
                Transpose(1, 2),
                nn.LayerNorm(decoder_dim, elementwise_affine=False),
                Transpose(1, 2),
                nn.GELU(),
            ]
            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else decoder_dim)
                for i in range(decoder_layers)
            ]
        )

        projs = []
        curr_dim = decoder_dim
        for i in range(proj_layers - 1):
            next_dim = int(curr_dim * proj_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask_info):
        # need x shape: [B, C, L]
        # x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        # predictor
        # need shape: [B, L, C] -> [B, L, D]
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if (
            residual is None
            or not self.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret
