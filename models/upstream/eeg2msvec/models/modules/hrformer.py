#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, China University of Petroleum
@Contact : zhoutao@s.upc.edu.cn
@Software: Visual Studio Code
@File    : hrformer.py
@Time    : 2024/05/22 22:04:11
@Desc    : 
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple
# from transformer import TransformerEncoderLayer
# from utils import Transpose

from models.modules.transformer import TransformerEncoderLayer
from models.modules.utils import Transpose



class ConvLn(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            padding: int = 0,
            groups: int = 1,
            dropout_prob: float = 0.0,
            bias: bool = False,
        ) -> None:
        super().__init__()
        def make_conv():
            conv = nn.Conv1d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=k,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            )
            nn.init.kaiming_normal_(conv.weight)
            return conv
        if dropout_prob == 0.0:
            self.conv_layer = nn.Sequential(
                make_conv(),
                nn.Sequential(
                    # transpose -2 and -1: 
                    # BxCxT -> BxTxC
                    Transpose(1, 2),
                    # if elementwise_affine is True, it add learnable parameters weight and bias for affine transformation
                    # after means normalizing the input data to mean 0 and variance 1, 
                    # the normalized input will be multiplied weight and added the bias.
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    Transpose(1, 2),
                ),
            )
        else:
            self.conv_layer = nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout_prob),
                nn.Sequential(
                    # transpose -2 and -1: 
                    # BxCxT -> BxTxC
                    Transpose(1, 2),
                    # if elementwise_affine is True, it add learnable parameters weight and bias for affine transformation
                    # after means normalizing the input data to mean 0 and variance 1, 
                    # the normalized input will be multiplied weight and added the bias.
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    Transpose(1, 2),
                ),
            )
    
    def forward(self, x):
        return self.conv_layer(x)


class ConvLnGeLU(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            padding: int = 0,
            groups: int = 1,
            dropout_prob: float = 0.0,
            bias: bool = False,
        ) -> None:
        super().__init__()
        def make_conv():
            conv = nn.Conv1d(
                in_channels=n_in,
                out_channels=n_out,
                kernel_size=k,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias
            )
            nn.init.kaiming_normal_(conv.weight)
            return conv
        if dropout_prob == 0.0:
            self.conv_layer = nn.Sequential(
                make_conv(),
                nn.Sequential(
                    # transpose -2 and -1: 
                    # BxCxT -> BxTxC
                    Transpose(1, 2),
                    # if elementwise_affine is True, it add learnable parameters weight and bias for affine transformation
                    # after means normalizing the input data to mean 0 and variance 1, 
                    # the normalized input will be multiplied weight and added the bias.
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    Transpose(1, 2),
                ),
                nn.GELU()
            )
        else:
            self.conv_layer = nn.Sequential(
                make_conv(),
                nn.Dropout(p=dropout_prob),
                nn.Sequential(
                    # transpose -2 and -1: 
                    # BxCxT -> BxTxC
                    Transpose(1, 2),
                    # if elementwise_affine is True, it add learnable parameters weight and bias for affine transformation
                    # after means normalizing the input data to mean 0 and variance 1, 
                    # the normalized input will be multiplied weight and added the bias.
                    nn.LayerNorm(n_out, elementwise_affine=True),
                    Transpose(1, 2),
                ),
                nn.GELU()
            )
    
    def forward(self, x):
        return self.conv_layer(x)


class TemporalFeatureExtractor(nn.Module):
    def __init__(
            self,
            feat_enc_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            conv_bias: bool = False
        ) -> None:
        super().__init__()
        assert isinstance(feat_enc_layers, list), \
            f"Input feature extractor config must be list. while provider: {type(feat_enc_layers)}={feat_enc_layers}"
        self.feat_env_layers = feat_enc_layers
        layers = []
        # input channel must be 1
        n_in = 1
        for _, feat_layer in enumerate(self.feat_env_layers):
            f_dim, f_k, f_s = feat_layer
            layers.append(
                ConvLnGeLU(n_in=n_in, n_out=f_dim, k=f_k, stride=f_s, dropout_prob=dropout, bias=conv_bias)
            )
            # the previous layer out_channel is equal to the next layer in_channel
            n_in = f_dim
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BxT -> BxCxT, C=1
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.layers(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: List[int],
            align_corners: bool = False,
            name: str = None
        ) -> None:
        """
        Donwsample module which generate the multi-stride the features based on previous output.
        :param int stride_pre: the previous stride
        :param int in_channel: the num of input channel
        :param List[int] strides: what strides which to generate 
        :param List[int] out_channels: the num of output channel
        :param bool align_corners: the parameter align_corners in F.interpolate, defaults to False
        :param str name: module name, defaults to None
        :raises ValueError: when the num of output is not equal to the length of list of output-stride

        Example:
            >>> tr = TransitionLayer(
                    in_channels=[64, 128]
                    out_channels=[64, 128, 256],
                    align_corners=False,
                    name=None
                )
                # for in_channel 64 and out_channel 64, 
                # if channels not equal, it will use conv1d(k=3,s=1,p=1) to map
                # while for the last out_channel 256, 
                # it will downsample from the last in_channel 128 by conv1d(k=3,s=2,p=1)
        """
        super(TransitionLayer, self).__init__()
        self.align_corners = align_corners
        self._num_out = len(out_channels)
        self._num_in = len(in_channels)
        if self._num_out != self._num_in + 1:
            raise ValueError(
                f"The length of `out_channels` does not equal to the length of `in_channels` + 1,"
                f"it generate {self._num_out} branches where provided {self._num_in}"
            )
        
        self.transition_layers = nn.ModuleList()
        for post_idx in range(self._num_out):
            # only downsample for the last in_channels
            if post_idx < self._num_in:
                if in_channels[post_idx] != out_channels[post_idx]:
                    layer = ConvLnGeLU(
                        n_in=in_channels[post_idx],
                        n_out=out_channels[post_idx],
                        k=3,
                        stride=1,
                        padding=1,
                    )
                else:
                    layer = None
            else:
                layer = ConvLnGeLU(
                    n_in=in_channels[-1],
                    n_out=out_channels[post_idx],
                    k=3,
                    stride=2,
                    padding=1,
                )
            self.transition_layers.append(layer)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for i, conv_bn_func in enumerate(self.transition_layers):
            if conv_bn_func is None:
                outs.append(x[i])
            else:
                out = conv_bn_func(x[-1])
                outs.append(out)
        return outs


class TransformerBranches(nn.Module):
    def __init__(
            self,
            num_blocks: List[int],
            # transformer encoder hypr-parameters
            embed_dims: List[int],
            ffn_embed_dims: List[int],
            num_attn_heads: List[int],
            dropout_probs: List[float],
            attn_dropout_probs: List[float],
            acti_dropout_probs: List[float],
            norm_firsts: List[bool] = [False, False],
            # module name
            name: str = None
        ) -> None:
        """
        the branches in the HighResolutionBlock.
        :param List[int] num_blocks: the number of transformer encoder layer block
        :param List[int] embed_dims: input feature dim
        :param List[int] ffn_embed_dims: the feedforward layer dim
        :param List[int] num_attn_heads: the num of attention head in MultiHeadAttention
        :param List[float] dropout_probs: the output dropout prob
        :param List[float] attn_dropout_probs: attention dropout prob in MultiHeadAttention
        :param List[float] acti_dropout_probs: activate dropout prob follow by activate_fn
        :param List[bool] norm_firsts: if True, it will use LayerNorm for normalize, defaults to [False, False]
        :param str name: the module name, defaults to None

        Example:
            >>> net = TransformerBranches(
                    num_blocks=[1, 1],
                    embed_dims=[64, 128],
                    ffn_embed_dims=[256, 512],
                    num_attn_heads=[1, 2],
                    dropout_probs=[0.1, 0.1],
                    attn_dropout_probs=[0.1, 0.1],
                    acti_dropout_probs=[0.1, 0.1],
                    norm_firsts=[False, False],
                    name="transformer_branch_1"
                )
        """
        super(TransformerBranches, self).__init__()

        self.transformer_branches = nn.ModuleList()
        # branch i
        for i in range(len(embed_dims)):
            transformer_encoder_layers = nn.ModuleList()
            # in branch i, it contain multi-block
            for j in range(num_blocks[i]):
                transformer_encoder_layers.add_module(
                    f"transformer_branch_{name}_{i+1}_{j+1}",
                    TransformerEncoderLayer(
                        embed_dim=embed_dims[i],
                        ffn_embed_dim=ffn_embed_dims[i],
                        num_attn_heads=num_attn_heads[i],
                        dropout_prob=dropout_probs[i],
                        attn_dropout_prob=attn_dropout_probs[i],
                        acti_dropout_prob=acti_dropout_probs[i],
                        norm_first=norm_firsts[i]
                    )
                )
            self.transformer_branches.append(transformer_encoder_layers)

    def forward(
            self,
            x: List[torch.Tensor],
            attn_mask: List[torch.Tensor] = None,
            key_padding_mask: List[torch.Tensor] = None,
            need_weights: bool = False
        ) -> List[torch.Tensor]:

        if attn_mask is None:
            attn_mask = [None] * len(x)
        if key_padding_mask is None:
            key_padding_mask = [None] * len(x)
        
        assert len(attn_mask) == len(x), f"Input attn_mask {len(attn_mask)} size must equal to x {len(x)}"
        assert len(key_padding_mask) == len(x), f"Input key_padding_mask {len(key_padding_mask)} size must equal to x {len(x)}"

        outs = []
        for idx, input in enumerate(x):
            # transformer encoder only process the time-length-axis: LxBxC

            # # shape: BxCxL -> BxLxC
            # conv = input.transpose(dim0=1, dim1=2)
            # # BxLxC -> LxBxC
            # conv = conv.transpose(dim0=0, dim1=1)

            # conv shape: BxCxL -> LxBxC
            conv = input.permute(2, 0, 1)
            for transformer_encoder_layer in self.transformer_branches[idx]:
                conv, (attn, layer_result) = transformer_encoder_layer(
                    conv, self_attn_mask=attn_mask[idx], self_attn_padding_mask=key_padding_mask[idx], need_weights=need_weights
                )
            # LxBxC -> BxLxC
            # conv = conv.transpose(dim0=0, dim1=1)
            # # BxLxC -> BxCxL
            # conv = conv.transpose(dim0=1, dim1=2)

            # conv shape: LxBxC -> BxCxL
            conv = conv.permute(1, 2, 0)
            outs.append(conv)
        return outs


class FuseLayers(nn.Module):
    def __init__(
            self,
            in_channels: List[int],
            out_channels: List[int],
            multi_scale_output: bool = True,
            align_corners: bool = False,
            name: str = None,
        ) -> None:
        """
        it fuse the multi-scale branches output features
        :param List[int] in_channels: the branches features need to fused channel list
        :param List[int] out_channels: the ouput channel list 
        :param bool multi_scale_output: if multi-scale_output is True, it will produce multi-scale features, defaults to True
        :param bool align_corners: the parameter align_corners in F.interpolate, defaults to False
        :param str name: module name, defaults to None
        Example:
            >>> net = FuseLayers(
                    in_channels=[64, 128],
                    out_channels=[64, 128, 256],
                    multi_scale_output=True,
                    align_corners=False,
                    name="fused_layer_1"
                )
        """
        super(FuseLayers, self).__init__()

        # the actual output num 
        self._actual_ch = len(out_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.fuse_layers = nn.Sequential()
        # for output i, it uses residual connection to process the input j
        for i in range(self._actual_ch):
            # generate the branch nn.ModuleList for every input j
            branch_fuse_layers = nn.ModuleList()
            # for input j
            for j in range(len(in_channels)):
                # for output i, it upsample the next-level j features for fusion
                if j > i:
                    branch_fuse_layers.append(
                        ConvLn(n_in=in_channels[j], n_out=out_channels[i], k=1, stride=1, padding=0,)
                    )
                # for output i, it downsample the previous-level j for fusion
                elif j < i:
                    # if  i - j > 0, it will do multi-downsample for fusion
                    # e.g. when i = 2, j = 0, it will twice-times downsample for fusion to generate branch output i=2
                    downsample = nn.Sequential()
                    for k in range(i - j - 1):
                        # if it isn't the last downsample block, it use conv-gn-gelu, without residual connection
                        mid_n_in = in_channels[j] * (2 ** k)
                        mid_n_out = out_channels[j] * (2 ** (k+1))
                        downsample.add_module(
                            f"{name}_branch{i+1}_input{j+1}_downsample{k+1}",
                            ConvLnGeLU(n_in=mid_n_in, n_out=mid_n_out, k=3, stride=2, padding=1,)
                        )
                    # the last downsample block, it will use skip connection follow by activation_fn
                    downsample.add_module(
                        f"{name}_branch{i+1}_input{j+1}_downsample{i-j}",
                        ConvLn(n_in=out_channels[j] * (2 ** (i-j-1)), n_out=out_channels[i], k=3, stride=2, padding=1,)
                    )
                    branch_fuse_layers.append(downsample)
                else:
                    branch_fuse_layers.add_module(f"residual_{name}_o{i+1}_i{j+1}", nn.Identity())
                # add to nn.Sequential
                self.fuse_layers.add_module(f"residual_fuse1_branch_{i+1}", branch_fuse_layers)
        if len(self.fuse_layers) == 0:
            self.fuse_layers.add_module(
                "identity", nn.Identity()
            )  

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for i in range(self._actual_ch):
            # for time-length-axis, it upsample
            residual_shape = x[i].shape[-1:]
            # i == j, it connect by identity 
            out = x[i]
            assert len(self._in_channels) == len(self.fuse_layers[i]), f"fuse layers for branch output {i} can not equal to in_channel"
            for j, fuse_layer in enumerate(self.fuse_layers[i]):
                if j > i:
                    y = fuse_layer(x[j])
                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='linear',
                        align_corners=self.align_corners)
                    out = out + y
                elif j < i:
                    y = fuse_layer(x[j])
                    out = out + y

            out = F.relu(out)
            outs.append(out)
        return outs


class HighResolutionModule(nn.Module):
    def __init__(
            self,
            num_blocks: List[int],
            in_channels: List[int],
            out_channels: List[int],
            # 
            ffn_embed_dims: List[int],
            num_attn_heads: List[int],
            dropout_probs: List[float],
            attn_dropout_probs: List[float],
            acti_dropout_probs: List[float],
            norm_firsts: List[bool] = [False, False],
            # 
            multi_scale_output: bool = True,
            align_corners: bool = False,
            name: str = None,
        ) -> None:
        """
        the HighResolutionBlock contains a Branches block and a FuseLayers
        :param List[int] num_blocks: the num blocks in Branches 
        :param List[int] in_channels: the input embedding-dim list, it rename embed_dims in MultiHeadAttention
        :param List[int] out_channels: the output embedding dim list
        :param List[int] ffn_embed_dims: the feedforward layer dim
        :param List[int] num_attn_heads: the num of attention head in MultiHeadAttention
        :param List[float] dropout_probs: the output dropout prob
        :param List[float] attn_dropout_probs: attention dropout prob in MultiHeadAttention
        :param List[float] acti_dropout_probs: activate dropout prob follow by activate_fn
        :param List[bool] norm_firsts: if True, it will use LayerNorm for normalize, defaults to [False, False]
        :param bool multi_scale_output: if multi-scale_output is True, it will produce multi-scale features, defaults to True
        :param bool align_corners: the parameter align_corners in F.interpolate, defaults to False
        :param str name: module name, defaults to None

        >>> net = HighResolutionModule(
                    num_blocks=[1, 1],
                    in_channels=[64, 128],
                    out_channels=[64, 128],
                    ffn_embed_dims=[256, 512],
                    num_attn_heads=[1, 2],
                    dropout_probs=[0.1, 0.1],
                    attn_dropout_probs=[0.1, 0.1],
                    acti_dropout_probs=[0.1, 0.1],
                    norm_firsts=[False, False],
                    multi_scale_output=True,
                    align_corners=False,
                    name="transformer_high_resolution_1"
                )
        """
        super(HighResolutionModule, self).__init__()

        self.branches = TransformerBranches(
            num_blocks=num_blocks,
            embed_dims=in_channels,
            ffn_embed_dims=ffn_embed_dims,
            num_attn_heads=num_attn_heads,
            dropout_probs=dropout_probs,
            attn_dropout_probs=attn_dropout_probs,
            acti_dropout_probs=acti_dropout_probs,
            norm_firsts=norm_firsts,
            name=name
        )

        self.fuse = FuseLayers(
            in_channels=in_channels,
            out_channels=out_channels,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners
        )

    def forward(
            self,
            x: List[torch.Tensor],
            attn_mask: List[torch.Tensor] = None,
            key_padding_mask: List[torch.Tensor] = None,
            need_weights: bool = False
        ) -> List[torch.Tensor]:
        # transformer encoder 
        out = self.branches(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights)
        out = self.fuse(out)
        return out


class Stage(nn.Module):
    def __init__(
        self,
        num_modules: int,
        num_blocks: List[int],
        in_channels: List[int],
        out_channels: List[int],
        ffn_embed_dims: List[int],
        num_attn_heads: List[int],
        dropout_probs: List[float],
        attn_dropout_probs: List[float],
        acti_dropout_probs: List[float],
        norm_firsts: List[bool],
        multi_scale_output: bool = True,
        align_corners: bool = False,
        name: str = None,
    ) -> None:
        """
        the u-hrformer stage block, a stage contains multi HighResolutionModule, 
        while a HighResolutionModule can also contain a TransformerBranches, which can contain multi-TransformerEncoderLayers
        :param int num_modules: the number of HighResolutionModule
        :param List[int] num_blocks: the number of TransformerEncoderLayer in the TransformerBranches, where it belongs to HighResolutionModule
        :param List[int] in_channels: the input channel list, it rename embed_dims in MultiHeadAttention
        :param List[int] out_channels: the output embedding dim list
        :param List[int] ffn_embed_dims: the feedforward layer dim
        :param List[int] num_attn_heads: the num of attention head in MultiHeadAttention
        :param List[float] dropout_probs: the output dropout prob
        :param List[float] attn_dropout_probs: attention dropout prob in MultiHeadAttention
        :param List[float] acti_dropout_probs: activate dropout prob follow by activate_fn
        :param List[bool] norm_firsts: if True, it will use LayerNorm for normalize
        :param bool multi_scale_output: if multi-scale_output is True, it will produce multi-scale features, defaults to True
        :param bool align_corners: the parameter align_corners in F.interpolate, defaults to False
        :param str name: the module name, defaults to None

        Example: 
            >>> stage_1 = Stage(
                num_modules=[1],
                num_blocks=[1],
                in_channels=[64, 128],
                out_channels=[64, 128],
                ffn_embed_dims=[256, 512],
                num_attn_heads=[1, 2],
                dropout_probs=[0.1, 0.1],
                attn_dropout_probs=[0.1, 0.1],
                acti_dropout_probs=[0.1, 0.1],
                norm_firsts=[False, False],
                multi_scale_output=True,
                align_corners=False,
                name="stage1",
            )
        """
        super(Stage, self).__init__()

        # the number of the HighResolutionModule in this Stage
        self._num_modules = num_modules

        self.stage_layers = nn.Sequential()
        for i in range(self._num_modules):
            # final HighResolutionModule and multi_scale_output is False, it will produce a single scale output
            if i == num_modules - 1 and not multi_scale_output:
                self.stage_layers.add_module(
                    f"stage_{name}_{i+1}",
                    HighResolutionModule(
                        num_blocks=num_blocks,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ffn_embed_dims=ffn_embed_dims,
                        num_attn_heads=num_attn_heads,
                        dropout_probs=dropout_probs,
                        attn_dropout_probs=attn_dropout_probs,
                        acti_dropout_probs=acti_dropout_probs,
                        norm_firsts=norm_firsts,
                        multi_scale_output=False,
                        align_corners=align_corners,
                        name=f"{name}_{i + 1}"
                    )
                )
            else:
                self.stage_layers.add_module(
                    f"stage_{name}_{i+1}",
                    HighResolutionModule(
                        num_blocks=num_blocks,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ffn_embed_dims=ffn_embed_dims,
                        num_attn_heads=num_attn_heads,
                        dropout_probs=dropout_probs,
                        attn_dropout_probs=attn_dropout_probs,
                        acti_dropout_probs=acti_dropout_probs,
                        norm_firsts=norm_firsts,
                        multi_scale_output=True,
                        align_corners=align_corners,
                        name=f"{name}_{i + 1}"
                    )
                )
        
    def forward(
            self, 
            x: List[torch.Tensor],
            attn_mask: List[torch.Tensor] = None,
            key_padding_mask: List[torch.Tensor] = None,
            need_weights: bool = False
        ) -> List[torch.Tensor]:
        out = x
        for idx in range(self._num_modules):
            out = self.stage_layers[idx](out, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=need_weights)
        return out



ARCHITECTURE_CONF = {
    "base": {
        "num_modules":          [1, 1, 1, 1],
        "in_channels":          [[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]],
        "num_blocks":           [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]],
        "ffn_embed_dims":       [[256], [256, 512], [256, 512, 1024], [256, 512, 1024, 2048]],
        "num_attn_heads":       [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]],
        "dropout_probs":        [[0.1], [0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        "attn_dropout_probs":   [[0.1], [0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        "acti_dropout_probs":   [[0.1], [0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
        "norm_first":           [[False], [False, False], [False, False, False], [False, False, False, False]],
    }
}


class HRFormerEncoder(nn.Module):
    def __init__(
        self,
        num_modules: List[int],
        in_channels: List[List[int]],
        num_blocks: List[List[int]],
        ffn_embed_dims: List[List[int]],
        num_attn_heads: List[List[int]],
        dropout_probs: List[List[float]],
        attn_dropout_probs: List[List[float]],
        acti_dropout_probs: List[List[float]],
        norm_first: List[List[bool]],
        fused_embed_dim: int = 256,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        assert len(num_modules) == len(in_channels)  == len(num_blocks) == \
            len(ffn_embed_dims) == len(num_attn_heads) == len(dropout_probs) == \
            len(attn_dropout_probs) == len(acti_dropout_probs) == len(norm_first), "please ensure the model config correct"

        def make_stage(idx: int, name: str, multi_scale_output: bool = True, align_corners: bool = False) -> nn.Module:
            assert idx < len(num_modules), "input idx can not larger than num_modules etc."
            return Stage(
                    num_modules=num_modules[idx],
                    num_blocks=num_blocks[idx],
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx],
                    ffn_embed_dims=ffn_embed_dims[idx],
                    num_attn_heads=num_attn_heads[idx],
                    dropout_probs=dropout_probs[idx],
                    attn_dropout_probs=attn_dropout_probs[idx],
                    acti_dropout_probs=acti_dropout_probs[idx],
                    norm_firsts=norm_first[idx],
                    multi_scale_output=multi_scale_output,
                    align_corners=align_corners,
                    name=name
                )
        
        def make_transition(idx: int, name: str):
            assert idx < len(num_modules) - 1, f"Can only build {len(num_modules) - 1} transition layers, while {idx} is larger than {len(num_modules) - 1}"
            return TransitionLayer(
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx+1],
                    align_corners=False,
                    name=name
                )
        
        # Architecture：
        #       Stage 1 -> Transition 1 
        #    -> Stage 2 -> Transition 2 
        #    -> Stage 3 -> Transition 3 
        #    -> Stage 4
        #    -> upsample -> Multi-Scale Fusion

        # stage 
        self.st1 = make_stage(0, name=f"st1")
        self.st2 = make_stage(1, name=f"st2")
        self.st3 = make_stage(2, name=f"st3")
        self.st4 = make_stage(3, name=f"st4")

        # transition
        self.tr1 = make_transition(0, name="tr1")
        self.tr2 = make_transition(1, name="tr2")
        self.tr3 = make_transition(2, name="tr3")

        # multi-scale fusion
        self.final_proj_layers = nn.ModuleList()
        self.fused_embed_dim = fused_embed_dim
        for i in range(len(in_channels)):
            self.final_proj_layers.append(
                nn.Sequential(
                    # nn.Linear(in_features=in_channels[i], out_features=D),
                    nn.Conv1d(in_channels=in_channels[-1][i], out_channels=self.fused_embed_dim, kernel_size=1, stride=1, bias=False),
                    Transpose(1, 2),
                    nn.LayerNorm(self.fused_embed_dim),
                    Transpose(1, 2),
                    nn.GELU(),
                )
            )
        # shape: BxCxL
        self.final_fused_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.fused_embed_dim * len(in_channels),
                out_channels=self.fused_embed_dim,
                kernel_size=1,
                stride=1
            ),
            Transpose(1, 2),
            nn.LayerNorm(self.fused_embed_dim),
            Transpose(1, 2),
        )


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.BoolTensor = None,
        key_padding_mask: torch.BoolTensor = None,
        need_branches: bool = False
    ) -> torch.Tensor:

        B, _, T = x.shape

        if key_padding_mask is None:
            key_padding_mask_lst = [None] * 4
        else:
            key_padding_mask_lst = self._key_padding_masks(
                key_padding_mask=key_padding_mask,
                raw_shape=[B, T],
            )

        # stage 1 & transition 1
        st1 = self.st1(
            [x],
            attn_mask=attn_mask,
            key_padding_mask=[key_padding_mask_lst[0]]
        )
        tr1 = self.tr1(st1)
        
        # stage 2 & transition 2
        st2 = self.st2(
            tr1,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst[0:2]
        )
        tr2 = self.tr2(st2)

        # stage 3 & transition 3
        st3 = self.st3(
            tr2,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst[0:3]
        )
        tr3 = self.tr3(st3)

        # stage 4
        st4 = self.st4(
            tr3,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst
        )

        outs = st4

        
        # upsampling time-dimension, BxCx(L/2) -> BxCxL
        T = (outs[0]).shape[-1:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(
                outs[i],
                size=T,
                mode='linear',
                align_corners=self.align_corners
            )

        # avgpooling on the embedding (channel) dim
        # shape: BxCxL
        # for i in range(len(outs)):
            # outs[i] = F.avg_pool2d(outs[i].unsqueeze(1), kernel_size=(2, 1), stride=(2, 1)).squeeze(1)
        for i in range(len(outs)):
            outs[i] = self.final_proj_layers[i](outs[i])
        out = torch.concat(outs, axis=1)
        # out = self.final_fused_layer(out)

        if need_branches:
            return out, outs
        return out, None

    def _compute_downsampled_data_length(
            self,
            raw_length: torch.Tensor | int,
            k: int = 3,
            s: int = 1,
            p: int = 0,
        ) -> torch.Tensor:
        """
        Compute the downsampled valid data length after applying convolution-like operations
        :param torch.Tensor | int raw_length: the raw input valid data length
        :param int k: downsampled kernel size, defaults to 3
        :param int s: downsampled kernel stride, defaults to 1
        :param int p: downsampled padding, defaults to 0
        :return torch.Tensor: the valid data length when downsampled
        """
        if isinstance(raw_length, torch.Tensor):
            return torch.floor((2 * p + raw_length - k) / s + 1).to(torch.long)
        else:
            return (2 * p + raw_length - k) // s + 1

    def _compute_key_padding_mask(
            self,
            raw_shape: Tuple[int, int],
            key_padding_mask: torch.BoolTensor,
            downsampled_layers: List[Tuple[int, int, int]],
            need_per_layer: bool = False,
        ) -> List[torch.BoolTensor]:
        """
        compute the downsampled key_padding_mask 
        :param Tuple[int] raw_shape: the raw input data shape (B, L)
        :param torch.BoolTensor key_padding_mask: the raw key_padding_mask, `True` means masking
        :param List[Tuple[int, int, int]] downsample_layers: the downsample module config, e.g. [(k, s, p), (...)]
        :param bool need_per_layer: wheather reserve the each key_padding_mask of per downsample layers
        :return List[torch.BoolTensor] : the downsample key_padding_mask list
        """
        def apply_downsample_to_key_padding_mask(
                key_padding_mask: torch.BoolTensor,
                true_length: torch.Tensor | int,
                B: int,
                L: int
            ) -> torch.BoolTensor:
            
            key_padding_mask = torch.zeros(size=(B, L))
            # 得到当前的所有掩码位置分界矩阵，即当序列长度 10，实际的数据长度为 7，则在矩阵的第 7 个元素为 1
            key_padding_mask[(
                torch.arange(key_padding_mask.shape[0], device=key_padding_mask.device),
                true_length - 1,
            )] = 1
            # 原始 wave 其 padding补充 0 的位置，在经过 CNN 后在特征图上的那些位置需要掩码用于输入到 Transformer 中
            # flip张量反转，在时间维度，则 padding mask 按照时间逆序
            # padding_mask.flip([-1]) 得到所有需要掩码分界张量（时间逆序），分界的位置值为 1
            # cumsum(-1)，在时间逆序的未掩码张量上进行求和，即第 n 项为其前 n 项的和。
            key_padding_mask = (1 - key_padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
            return key_padding_mask
        
        key_padding_mask_lst = []
        if key_padding_mask is not None and key_padding_mask.any():
            B, L = raw_shape
            # compute the undownsampled valid data length in the batched key_padding_mask
            true_length = (1 - key_padding_mask.long()).sum(-1)
            
            num_down_layers = len(downsampled_layers)
            # compute the downsampled_key_padding_mask
            for idx, (k, s, p) in enumerate(downsampled_layers):
                # compute the output valid data length of the current downsampled module
                true_length = self._compute_downsampled_data_length(true_length, k=k, s=s, p=p)
                # compute the output data length of the current downsampled module
                L = self._compute_downsampled_data_length(L, k=k, s=s, p=p)
                if need_per_layer:
                    key_padding_mask_lst.append(
                        apply_downsample_to_key_padding_mask(
                            key_padding_mask, true_length, B, L
                        )
                    )
            if not need_per_layer:
                key_padding_mask_lst.append(
                    apply_downsample_to_key_padding_mask(
                        key_padding_mask, true_length, B, L
                    )
                )
        return key_padding_mask_lst

    def _key_padding_masks(
            self,
            key_padding_mask: torch.BoolTensor,
            raw_shape: Tuple[int, int],
        ) -> List[torch.BoolTensor]:
        """
        compute all key_padding_mask of all downsampled layers
        :param torch.BoolTensor key_padding_mask: the raw key_padding_mask
        :param Tuple[int, int] raw_shape: the raw input data shape [B, L]
        :return List[torch.BoolTensor]: all downsampled key_padding_mask 
        """
        # for stage 1
        # feat_enc_layers = [(k, s, 0) for (_, k, s) in self.feat_enc_layers]
        # for stage 2-4
        tran_enc_layers = [(3, 2, 1) for _ in range(3)]
        # get key_padding_mask for the per downsample step
        key_padding_mask_lst = [key_padding_mask.clone()]
        key_padding_mask_lst += self._compute_key_padding_mask(
            raw_shape,
            key_padding_mask,
            tran_enc_layers,
            need_per_layer=True
        )
        key_padding_mask_lst = key_padding_mask_lst[-4:]
        return [mask.to(key_padding_mask.device) for mask in key_padding_mask_lst]


class HRFormer(HRFormerEncoder):
    def __init__(
        self,
        num_modules: List[int],
        in_channels: List[List[int]],
        num_blocks: List[List[int]],
        ffn_embed_dims: List[List[int]],
        num_attn_heads: List[List[int]],
        dropout_probs: List[List[float]],
        attn_dropout_probs: List[List[float]],
        acti_dropout_probs: List[List[float]],
        norm_first: List[List[bool]],

        feat_enc_layers: List[Tuple[int, int, int]],
        feat_dropout_prob: float = 0.0,
        feat_enc_bias: bool = False,

        align_corners: bool = False,
    ) -> None:
        super(HRFormer, self).__init__(
            num_modules,
            in_channels,
            num_blocks,
            ffn_embed_dims,
            num_attn_heads,
            dropout_probs,
            attn_dropout_probs,
            acti_dropout_probs,
            norm_first
        )
        self.align_corners = align_corners
        self.feat_enc_layers = feat_enc_layers

        assert len(num_modules) == len(in_channels)  == len(num_blocks) == \
            len(ffn_embed_dims) == len(num_attn_heads) == len(dropout_probs) == \
            len(attn_dropout_probs) == len(acti_dropout_probs) == len(norm_first), "please ensure the model config correct"

        # stem
        self.feature_extractor = TemporalFeatureExtractor(
            feat_enc_layers=feat_enc_layers, dropout=feat_dropout_prob, conv_bias=feat_enc_bias
        )

        def make_stage(idx: int, name: str, multi_scale_output: bool = True, align_corners: bool = False) -> nn.Module:
            assert idx < len(num_modules), "input idx can not larger than num_modules etc."
            return Stage(
                    num_modules=num_modules[idx],
                    num_blocks=num_blocks[idx],
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx],
                    ffn_embed_dims=ffn_embed_dims[idx],
                    num_attn_heads=num_attn_heads[idx],
                    dropout_probs=dropout_probs[idx],
                    attn_dropout_probs=attn_dropout_probs[idx],
                    acti_dropout_probs=acti_dropout_probs[idx],
                    norm_firsts=norm_first[idx],
                    multi_scale_output=multi_scale_output,
                    align_corners=align_corners,
                    name=name
                )
        
        def make_transition(idx: int, name: str):
            assert idx < len(num_modules) - 1, f"Can only build {len(num_modules) - 1} transition layers, while {idx} is larger than {len(num_modules) - 1}"
            return TransitionLayer(
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx+1],
                    align_corners=False,
                    name=name
                )
        
        # stage 
        self.st1 = make_stage(0, name=f"st1")
        self.st2 = make_stage(1, name=f"st2")
        self.st3 = make_stage(2, name=f"st3")
        self.st4 = make_stage(3, name=f"st4")

        # transition
        self.tr1 = make_transition(0, name="tr1")
        self.tr2 = make_transition(1, name="tr2")
        self.tr3 = make_transition(2, name="tr3")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.BoolTensor = None,
        key_padding_mask: torch.BoolTensor = None,
        need_branches: bool = False
    ) -> torch.Tensor:

        if key_padding_mask is None:
            key_padding_mask = [None] * 4
        B, _, T = x.shape
        key_padding_mask_lst = self._key_padding_mask(
            key_padding_mask=key_padding_mask,
            raw_shape=[B, T],
            device=x.device
        )

        # downsample by 1-d CNN
        features = self.feature_extractor(x)

        # stage 1 & transition 1
        st1 = self.st1(
            [features],
            attn_mask=attn_mask,
            key_padding_mask=[key_padding_mask_lst[0]]
        )
        tr1 = self.tr1(st1)
        
        # stage 2 & transition 2
        st2 = self.st2(
            tr1,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst[0:2]
        )
        tr2 = self.tr2(st2)

        # stage 3 & transition 3
        st3 = self.st3(
            tr2,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst[0:3]
        )
        tr3 = self.tr3(st3)

        # stage 4
        st4 = self.st4(
            tr3,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_lst
        )

        outs = st4
        for i in range(len(outs)):
            # avgpooling on the embedding dim
            outs[i] = F.avg_pool2d(outs[i].unsqueeze(1), kernel_size=(2, 1), stride=(2, 1)).squeeze(1)                                                     

        # upsampling
        T = (outs[0]).shape[-1:]
        for i in range(1, len(outs)):
            outs[i] = F.interpolate(
                outs[i],
                size=T,
                mode='linear',
                align_corners=self.align_corners
            )
        out = torch.concat(outs, axis=1)

        if need_branches:
            return out, outs
        return out

    def _key_padding_mask(
            self,
            key_padding_mask: torch.BoolTensor,
            raw_shape: List[int],
            device: torch.device
        ) -> torch.BoolTensor:
        # for stage 1
        feat_enc_layers = [(k, s, 0) for (_, k, s) in self.feat_enc_layers]
        # for stage 2-4
        tran_enc_layers = [(3, 2, 1) for _ in range(3)]
        # get key_padding_mask for the per downsample step
        key_padding_mask_lst = self._compute_key_padding_mask(
            raw_shape,
            key_padding_mask,
            feat_enc_layers + tran_enc_layers,
            need_per_layer=True
        )
        key_padding_mask_lst = key_padding_mask_lst[-4:]
        return [mask.to(device) for mask in key_padding_mask_lst]


if __name__ == "__main__":
    # net = FuseLayers(
    #     in_channels=[64, 128],
    #     out_channels=[64, 128, 256],
    #     multi_scale_output=True,
    #     align_corners=False,
    #     name="fused_layer_1"
    # )
    # print(net)

    raw_x = torch.zeros(size=(2, 1, 1000000))
    key_padding_mask = torch.zeros(size=(2, 1000000), dtype=torch.int)
    key_padding_mask[0, 400000:] = 1
    key_padding_mask[1, 30000:] = 1
    raw_key_padding_mask = key_padding_mask.bool()

    conf = ARCHITECTURE_CONF.get("base")

    # HRFormerEncoder
    hrformer_enc = HRFormerEncoder(
        num_modules=conf.get("num_modules"),
        in_channels=conf.get("in_channels"),
        num_blocks=conf.get("num_blocks"),
        ffn_embed_dims=conf.get("ffn_embed_dims"),
        num_attn_heads=conf.get("num_attn_heads"),
        dropout_probs=conf.get("dropout_probs"),
        attn_dropout_probs=conf.get("attn_dropout_probs"),
        acti_dropout_probs=conf.get("acti_dropout_probs"),
        norm_first=conf.get("norm_first"),
    )
    print(hrformer_enc.to("cuda:0"))

    key_padding_mask = hrformer_enc._compute_key_padding_mask(
        raw_shape=(raw_x.shape[0], raw_x.shape[-1]),
        key_padding_mask=raw_key_padding_mask,
        downsampled_layers=[(32, 5, 0), (5, 2, 0), (3, 2, 0)],
        need_per_layer=False
    )[0]

    print(key_padding_mask.shape)
    stem = TemporalFeatureExtractor(
        feat_enc_layers=[(64, 32, 5), (64, 5, 2), (64, 3, 2)],
        dropout=0.0,
        conv_bias=False,
    )
    x = stem(raw_x)

    y, branches = hrformer_enc(
        x.to("cuda:0"),
        attn_mask=None,
        key_padding_mask=key_padding_mask.to("cuda:0"),
        need_branches=True
    )
    print(y.shape)
    print([t.shape for t in branches])


    # HRFormer
    # hrformer = HRFormer(
    #     num_modules=conf.get("num_modules"),
    #     in_channels=conf.get("in_channels"),
    #     num_blocks=conf.get("num_blocks"),
    #     ffn_embed_dims=conf.get("ffn_embed_dims"),
    #     num_attn_heads=conf.get("num_attn_heads"),
    #     dropout_probs=conf.get("dropout_probs"),
    #     attn_dropout_probs=conf.get("attn_dropout_probs"),
    #     acti_dropout_probs=conf.get("acti_dropout_probs"),
    #     norm_first=conf.get("norm_first"),
    #     feat_enc_layers=[(64, 32, 5), (64, 5, 2), (64, 3, 2)],
    #     feat_dropout_prob=0.0,
    #     feat_enc_bias=False,
    # )
    # print(hrformer)

    # y, branches = hrformer(
    #     raw_x,
    #     attn_mask=None,
    #     key_padding_mask=raw_key_padding_mask,
    #     need_branches=True
    # )
    # print(y.shape)
    # print([t.shape for t in branches])
    pass
