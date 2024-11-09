#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : transformer.py
@Time    : 2024/05/10 15:04:56
@Desc    : 
"""
import math
import torch
import warnings
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
try:
    from flash_attn import __version__ as flash_attn_version
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_kvpacked_func,
    )
except ImportError:
    print("flash_attn not installed, please install it first")
# from utils import SamePad, Transpose
from .utils import SamePad, Transpose


class ConvPositionEmbedding(nn.Module):
    def __init__(
            self, 
            conv_pos_dim: int,
            conv_pos_layers: int,
            conv_pos_groups: int
        ) -> None:
        """
        the conv position embedding block, it use the input [B, C, L] to gnereate the position embedding [B, C, L] 
        :param int conv_pos_dim: the input embedding dim
        :param int conv_pos_layers: the conv position embedding block layers
        :param int conv_pos_groups: use the grouped conv by the groups
        
        Examples:
            >>> net = ConvPositionEmbedding(128, 1, 16)
            >>> y = net(torch.zeros(size=(8, 128, 1000)))
            >>> y.shape is [8, 128, 1000]
        """

        super().__init__()

        def make_conv_module(
                embed_dim: int,
                kernel: int,
                groups: int,
                num_layers: int,
            ) -> nn.Sequential:
            layers = []
            for _ in range(num_layers):
                conv_layer = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel, padding=kernel // 2, groups=groups)
                # Initialize the convolution weights
                # typically set according to your training configuration
                dropout = 0
                std = math.sqrt((4 * (1.0 - dropout)) / (kernel * embed_dim))
                nn.init.normal_(conv_layer.weight, mean=0, std=std)
                nn.init.constant_(conv_layer.bias, 0)
                
                # when the conv position block has a conv layer, use weight norm
                if num_layers == 1:
                    # nn.utils.weight_norm is deprecated 
                    # conv_layer = nn.utils.weight_norm(conv_layer, name='weight', dim=2)
                    conv_layer = nn.utils.parametrizations.weight_norm(conv_layer, name='weight', dim=2)
                
                # Stack up the convolution, padding, and activation
                conv_seq = [conv_layer, SamePad(kernel), nn.GELU()]

                # when the conv position block has many conv layers, use LayerNorm instead of weight norm
                if num_layers > 1:
                    conv_seq.insert(2, Transpose(1, 2))
                    conv_seq.insert(3, nn.LayerNorm(embed_dim, elementwise_affine=False))
                    conv_seq.insert(4, Transpose(1, 2))

                layers.append(nn.Sequential(*conv_seq))

            return nn.Sequential(*layers)

        kernel = max(3, conv_pos_dim // conv_pos_layers)
        self.layers = make_conv_module(
            embed_dim=conv_pos_dim,
            kernel=kernel,
            groups=conv_pos_groups,
            num_layers=conv_pos_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.layers(x)
        return pos_embed


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first: bool = True
        ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.batch_first = batch_first
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5


        assert self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )
        # 
        self.dropout_module = nn.Dropout(p=dropout)

        self.bias = bias
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # initialized the input-projection layer weights
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.q_proj.bias, 0)

        # initialize the out projection layer weight & bias
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
    def forward(
            self,
            query,
            key, 
            value,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            need_weights: bool = False,
            need_avg_head_weights: bool = False
        ):

        is_batched = query.dim() == 3

        # if data shape is [B, L, D], transpose to [L, B, D]
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # query, key, value projection
        tgt_len, bsz, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        kv_bsz = bsz
        kv_bsz = k.size(1)
        k = (
            k.contiguous()
            .view(-1, kv_bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(-1, kv_bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        assert k is not None
        src_len = tgt_len
        assert k.size(1) == src_len

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len
        # attn probs
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights_float = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        
        attn = self.out_proj(attn)

        attn_weights = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if need_avg_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        
        # transpose to [B, L, D]
        if self.batch_first and is_batched:
            return attn.transpose(1, 0), attn_weights
        else:
            return attn, attn_weights


class PyMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False
        ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5


        assert self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )
        # 
        self.dropout_module = nn.Dropout(p=dropout)

        self.bias = bias
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        self._reset_parameters()

        self.skip_embed_dim_check = False

    def _reset_parameters(self):
        # initialized the input-projection layer weights
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        # if self.qkv_same_dim:
        #     nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        #     nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        #     nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        # else:
        #     nn.init.xavier_uniform_(self.k_proj.weight)
        #     nn.init.xavier_uniform_(self.v_proj.weight)
        #     nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.q_proj.bias, 0)

        # initialize the out projection layer weight & bias
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
        # initialize the bias parameters
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor],
            value: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        is_batched = query.dim() == 3

        # single direction info, historical info
        if is_causal:
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype
            )

            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

        # if data shape is [B, L, D], transpose to [L, B, D]
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # do multi head attention
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout_module.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )
        # transpose to [B, L, D]
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class FlashMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False
        ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** (-0.5)


        assert self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )
        self.dropout_module = nn.Dropout(p=dropout)

        self.bias = bias
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        self._reset_parameters()

        self.skip_embed_dim_check = False

    def _reset_parameters(self):
        # initialized the input-projection layer weights
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        # if self.qkv_same_dim:
        #     nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        #     nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        #     nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        # else:
        #     nn.init.xavier_uniform_(self.k_proj.weight)
        #     nn.init.xavier_uniform_(self.v_proj.weight)
        #     nn.init.xavier_uniform_(self.q_proj.weight)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.q_proj.bias, 0)

        # initialize the out projection layer weight & bias
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.constant_(self.o_proj.bias, 0.0)
        
        # initialize the bias parameters
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    def ori_forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor],
            value: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[torch.Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        is_batched = query.dim() == 3

        # single direction info, historical info
        if is_causal:
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype
            )

            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

        # if data shape is [B, L, D], transpose to [L, B, D]
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # do multi head attention
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout_module.p,
            out_proj_weight=self.o_proj.weight,
            out_proj_bias=self.o_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal
        )
        # transpose to [B, L, D]
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        FlashAttention forward func
        :param torch.Tensor hidden_states: the FlashMultiheadAttn input x
        :param Optional[torch.Tensor] key_padding_mask: which position will be ignore, defaults to None, 1 means to ignore, where 0 means to restore
        :param Optional[Tuple[torch.Tensor]] past_key_value: the cache key or value, defaults to None
        :param bool output_attentions: where output attention weights, defaults to False
        :param bool use_cache: whether the output the past_key_value, defaults to False
        :param bool is_causal: whether the causal attention, defaults to False
        :return Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: _description_
        """
        if output_attentions:
            warnings.warn(
                "Output attentions is not supported for FlashMultiheadAttention, returning `None` instead."
            )
        
        # get the batch size and seq length
        bsz, q_len, _ = hidden_states.size()

        kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

        # shape: (b, s, num_heads, head_dim)
        q, k, v = (
            op(hidden_states).view(bsz, q_len, nh, self.head_dim)
            for op, nh in (
                (self.q_proj, self.num_heads),
                (self.k_proj, kv_heads),
                (self.v_proj, kv_heads),
            )
        )

        kv_seq_len = k.shape[1]
        past_kv_len = 0
        if past_key_value is not None:
            past_kv_len = past_key_value[0].shape[2]
            kv_seq_len += past_kv_len

        if past_key_value is not None:
            assert (
                flash_attn_version >= "2.1.0"
            ), "past_key_value support requires flash-attn >= 2.1.0"

            # reuse k, v
            k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
            v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

        past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

        if key_padding_mask is None:
            output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=is_causal).view(
                bsz, q_len, -1
            )
        else:
            # the input key_padding_mask: 1 means to ignore, where in unpad_input, 1 means to reserve
            key_padding_mask = torch.logical_not(key_padding_mask)
            q, indices, cu_q_lens, max_s = unpad_input(q, key_padding_mask[:, -q_len:])
            # We can skip concat and call unpad twice but seems better to call unpad only once.
            kv, _, cu_k_lens, max_k = unpad_input(
                torch.stack((k, v), dim=2), key_padding_mask
            )
            output_unpad = flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_q_lens,
                cu_k_lens,
                max_s,
                max_k,
                0.0,
                softmax_scale=None,
                causal=is_causal
            )
            output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
            output = pad_input(output_unpad, indices, bsz, q_len)

        return self.o_proj(output), None, past_key_value


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: float = 768,
        ffn_embed_dim: float = 3072,
        num_attn_heads: int = 8,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        acti_dropout_prob: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        """
        the Transformer encoder layers.
        :param float embed_dim: the input data dim, defaults to 768
        :param float ffn_embed_dim: the feed forward hidden size, defaults to 3072
        :param int num_attn_heads: the multi-head-attention heads number, defaults to 8
        :param float dropout_prob: the output dropout prob, defaults to 0.1
        :param float attn_dropout_prob: the multi-head-attention attention dropout prob, defaults to 0.1
        :param float activation_dropout: the dropout module prob followed by activation fn, defaults to 0.1
        :param bool norm_first: where use norm firstly, defaults to False

        Example:
            >>> encoder = TransformerEncoderLayer(
                embed_dim=128,
                ffn_embed_dim=512,
                num_attn_heads=8,
                dropout_prob=0.1,
                attn_dropout_prob=0.1,
                acti_dropout_prob=0.1
            )
        """
        super().__init__()
        # Initialize parameters
        self.embed_dim = embed_dim
        self.dropout_prob = dropout_prob
        self.acti_dropout_prob = acti_dropout_prob
        self.norm_first = norm_first

        # Initialize blocks
        self.activation_fn = nn.GELU(approximate='none')
        # use multi head attention
        # self.self_attn = MultiHeadAttention(
        self.self_attn = PyMultiHeadAttention(
            self.embed_dim,
            num_attn_heads,
            dropout=attn_dropout_prob,
            batch_first=False
        )

        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(self.acti_dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)

        # layer norm associated with the self attention layer
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)

        # layer norm associated with the position wise feed-forward NN
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.norm_first:
            x = self.norm1(x)
            # multi-head-attention
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=need_weights,
            )
            x = residual + self.dropout1(x)
            residual = x

            x = self.norm2(x)
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x
            x = residual + self.dropout3(x)
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )
            x = residual + self.dropout1(x)
            x = self.norm1(x)

            residual = x
            # fc1 -> activate(GELU/ReLU) -> dropout -> fc2
            # x = self.fc2(self.dropout2(self.activation_fn(self.fc1(x))))
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = residual + self.dropout3(x)
            x = self.norm2(x)

        return x, (attn, layer_result)


if __name__ == "__main__":
    # x = torch.ones(size=(8, 1000, 512))
    # key_padding_mask = torch.zeros(size=(8, 1000), dtype=torch.int)
    # key_padding_mask[0, 800:] = 1
    # key_padding_mask[1, 600:] = 1
    # key_padding_mask = key_padding_mask.bool()
    # mha = MultiheadAttention(
    #     embed_dim=512, num_heads=8, dropout=0, batch_first=True
    # )
    # y1, attn = mha(
    #     query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=None
    # )
    # print(y1.shape, attn.shape)

    # device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    # x = torch.ones(size=(128, 1250, 512), dtype=torch.float16)
    # key_padding_mask = torch.zeros(size=(512, 1250), dtype=torch.int)
    # key_padding_mask[0, 800:] = 1
    # key_padding_mask[1, 600:] = 1
    # # key_padding_mask[0, :800] = 1
    # # key_padding_mask[1, :600] = 1
    # key_padding_mask = key_padding_mask.bool()

    # fmha = FlashMultiheadAttention(
    #     embed_dim=512, num_heads=8, dropout=0, batch_first=True
    # ).to(device).half()
    # x = x.to(device)
    # key_padding_mask = key_padding_mask.to(device)

    # fmha.eval()
    # with torch.no_grad():
    #     key_padding_mask = None
    #     # during in the 
    #     # flash_key_padding_mask = ~key_padding_mask
    #     iter_num = 100
    #     import time
    #     start_time = time.time()
    #     for _ in range(iter_num):
    #         y = x
    #         y, attn, _ = fmha(
    #             hidden_states=y, key_padding_mask=key_padding_mask
    #         )
    #     end_time = time.time()
    #     print(f"FlashAttn: {(end_time - start_time) / iter_num}")

    #     start_time = time.time()
    #     for _ in range(iter_num):
    #         ori_y = x
    #         ori_y, attn = fmha.ori_forward(
    #             query=ori_y, key=ori_y, value=ori_y, key_padding_mask=key_padding_mask, attn_mask=None
    #         )
    #     end_time = time.time()
    #     print(f"Pytorch: {(end_time - start_time) / iter_num}")

    net = ConvPositionEmbedding(128, 1, 16)
    y = net(torch.zeros(size=(8, 128, 1000)))
    print(y.shape)


    x = torch.ones(size=(2, 1250, 512))
    key_padding_mask = torch.zeros(size=(2, 1250), dtype=torch.int)
    key_padding_mask[0, 800:] = 1
    key_padding_mask[1, 600:] = 1

    mha = MultiHeadAttention(
        embed_dim=512, num_heads=8, dropout=0
    )
    mha.eval()
    with torch.no_grad():
        y, _ = mha(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=None
        )
    
    pymha = PyMultiHeadAttention(
        embed_dim=512, num_heads=8, dropout=0, batch_first=True
    )
    pymha.eval()
    key_padding_mask = key_padding_mask.bool()
    with torch.no_grad():
        y2, _ = pymha(
            query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=None
        )

    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    fmha = FlashMultiHeadAttention(
        embed_dim=512, num_heads=8, dropout=0, batch_first=True
    ).to(device).half()
    fmha.eval()
    with torch.no_grad():
        y3, attn, _ = fmha(
            hidden_states=x.half().to(device), key_padding_mask=key_padding_mask.to(device)
        )
        y3_1, attn_1 = fmha.ori_forward(
            query=x.half().to(device), key=x.half().to(device), value=x.half().to(device), key_padding_mask=key_padding_mask.to(device)
        )
    print()
    pass
