#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : beliefppg.py
@Time    : 2025/03/29 20:00:08
@Desc    : 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.beliefppg.prior_layer import PriorLayer


class AveragePooling1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class CausalConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=None,
            dilation=1,
            groups=1,
            bias=True
        ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation
        else:
            assert padding == (kernel_size - 1) * dilation, "padding must be equal to (kernel_size - 1) * dilation"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.chomp = Chomp1d(padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.chomp(x)
        return x


class TimeDomainBackbone(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        # Block 1
        # 因果卷积padding计算：(kernel_size-1)*dilation
        self.conv1 = CausalConv1d(1, 16, kernel_size=10, stride=1, dilation=2, padding=18) 
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Block 2
        self.conv2 = CausalConv1d(16, 16, kernel_size=10, stride=1, dilation=2, padding=18)
        self.bn2 = nn.BatchNorm1d(16)
        self.drop2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # LSTM layers
        # self.lstm1 = nn.LSTM(16, 64, num_layers=1, batch_first=True, dropout=0.1)
        # self.drop3 = nn.Dropout1d(p=0.1)
        # self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True, dropout=0.1)
        # self.drop4 = nn.Dropout1d(p=0.1)
        self.lstm = nn.LSTM(16, 64, num_layers=2, batch_first=True, dropout=0.1)
        
        # Final dense layers
        self.feat_branch = nn.Linear(64, output_shape)
        self.value_branch = nn.Linear(64, output_shape)
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        # Input shape: (batch, seq_len, 1)
        x = x.permute(0, 2, 1)  # (batch, 1, seq_len)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.leaky_relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.leaky_relu(x)
        x = self.pool2(x)
        
        # LSTM处理
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        # x, (h_n, c_n) = self.lstm1(x)
        # x = self.drop3(x)
        # x, (h_n, c_n) = self.lstm2(x, (h_n, c_n))  # 输出形状 (batch, seq_len, 64)
        # x = self.drop4(x)
        x, (h_n, c_n) = self.lstm(x)  # 取LSTM的输出
        x = x[:, -1, :]  # 取最后一个时间步
        
        # 分支输出
        feat = self.leaky_relu(self.feat_branch(x))
        value = self.leaky_relu(self.value_branch(x))
        return feat, value


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            drop_prob=0.0,
            bias=True,
            kdim=None,
            vdim=None,
            qk_norm: bool = False,
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
        self.dropout = nn.Dropout(p=drop_prob)

        self.bias = bias
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if qk_norm:
            self.q_norm = nn.LayerNorm(embed_dim)
            self.k_norm = nn.LayerNorm(embed_dim)
        else:
            self.q_norm = self.k_norm = None

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

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

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
        attn_probs = self.dropout(attn_weights)
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


class PositionalEncoding(nn.Module):
    def __init__(self, seqlen, d_model, num_dims=3):
        """
        PyTorch 版位置编码实现
        :param seqlen: 序列长度 (n_blocks)
        :param d_model: 编码维度 (需与输入维度匹配)
        :param num_dims: 输入维度数 (3或4)
        """
        super().__init__()
        self.seqlen = seqlen
        self.d_model = d_model
        self.num_dims = num_dims
        
        # 预计算位置编码
        pe = self._positional_encoding(seqlen, d_model)
        self.register_buffer('pe', pe)  # 注册为缓冲区
        
    def _positional_encoding(self, length, depth):
        """生成位置编码矩阵"""
        depth = depth / 2
        
        # 使用 PyTorch 生成位置编码
        positions = torch.arange(length).unsqueeze(1)  # (seqlen, 1)
        depths = torch.arange(depth).unsqueeze(0) / depth  # (1, depth)
        
        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (seqlen, depth)
        
        # 拼接正弦和余弦部分
        pe = torch.cat([
            torch.sin(angle_rads),
            torch.cos(angle_rads)
        ], dim=-1)  # (seqlen, d_model)
        
        return pe.float()  # 确保数据类型为 float32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入张量，形状为:
            - num_dims=3: (batch_size, seqlen, d_model)
            - num_dims=4: (batch_size, *, seqlen, d_model)
        """
        # 根据输入维度添加位置编码
        if x.dim != self.num_dims:
            dim = x.dim
        else:
            dim = self.num_dims
        if dim == 3:
            # 输入形状 (batch, seqlen, features)
            x_pos = self.pe.unsqueeze(0)  # (1, seqlen, d_model)
        else:
            # 输入形状 (batch, *, seqlen, features)
            x_pos = self.pe.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, d_model)
            
        return x_pos


class DoubleAxisAttention(nn.Module):
    def __init__(self, channels, n_frames, n_bins):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        self.pos_freq = PositionalEncoding(seqlen=n_bins, d_model=channels)
        self.pos_time = PositionalEncoding(seqlen=n_frames, d_model=channels)
        self.attn_freq = MultiHeadAttention(embed_dim=channels, num_heads=1, batch_first=True)
        self.attn_time = MultiHeadAttention(embed_dim=channels, num_heads=1, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, frames, bins, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (batch, 2, frames, bins)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (batch, frames, bins, channels)

        # shape 
        batch, frames, bins, embed_dim = x.shape
        
        # Frequency attention
        # (batch, frames, bins, embed_dim)
        x_freq = x + self.pos_freq(x)
        x_freq = x_freq.view(-1, bins, embed_dim)
        freq_attn, _ = self.attn_freq(x_freq, x_freq, x_freq)
        freq_attn = freq_attn.view(batch, frames, bins, embed_dim)
        
        # Time attention
        # (batch, bins, frames, ch)
        x_time = x.permute(0, 2, 1, 3).contiguous()
        x_time = x_time + self.pos_time(x_time)
        x_time = x_time.view(-1, frames, embed_dim)
        time_attn, _ = self.attn_time(x_time, x_time, x_time)
        time_attn = time_attn.view(batch, bins, frames, embed_dim).permute(0, 2, 1, 3)
        
        return time_attn, freq_attn


class UpsampleFusion(nn.Module):
    def __init__(
        self,
        x_channel: int,
        y_channel: int,
        inter_channel: int,
        factor: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor)

        self.conv1 = nn.Conv1d(y_channel, inter_channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(x_channel, inter_channel, kernel_size=1, stride=1)
        self.w_conv = nn.Conv1d(inter_channel, 1, kernel_size=1, stride=1)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)

        theta_y = self.conv1(y)
        phi_up = self.conv2(up)
        psi_f = F.relu(theta_y + phi_up)
        psi_f = self.w_conv(psi_f)
        rate = torch.sigmoid(psi_f)
        att_y = y * rate

        # concat on channel dimension
        out = torch.concat([up, att_y], dim=1)
        return out


class HybridUNet(nn.Module):
    def __init__(
            self,
            depth: int = 3,
            attn_channels: int = 32,
            init_channels: int = 12,
            down_fac: int = 4,
            n_frames: int = 7,
            n_bins: int = 64,
            use_time_backbone: bool = True
        ) -> None:
        super().__init__()
        self.depth = depth
        self.down_fac = down_fac
        self.use_time_backbone = use_time_backbone
        
        # Initial blocks
        self.double_attn = DoubleAxisAttention(attn_channels, n_frames, n_bins)
        self.avg_pool = AveragePooling1D(1)
        self.dense = nn.Linear(2, attn_channels)

        if use_time_backbone:
            time_embed_dim = init_channels * (2 ** (depth - 1))
            self.time_backbone = TimeDomainBackbone(time_embed_dim)
            self.weight_conv = nn.Sequential(
                nn.Conv1d(time_embed_dim, time_embed_dim, kernel_size=2),
                nn.Tanh(),
                nn.Dropout(0.2)
            )
            self.feat_conv = nn.Sequential(
                nn.Conv1d(time_embed_dim, time_embed_dim, kernel_size=2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

        # Downsample blocks
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = init_channels
        for idx in range(depth):
            if idx == 0:
                self.down_convs.append(
                    nn.Sequential(
                        nn.Conv1d(attn_channels, channels, 3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Conv1d(channels, channels, 3, padding=1),
                        nn.ReLU()
                    )
                )
            else:
                self.down_convs.append(
                    nn.Sequential(
                        nn.Conv1d(channels // 2, channels, 3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Conv1d(channels, channels, 3, padding=1),
                        nn.ReLU()
                    )
                )
            self.pools.append(nn.MaxPool1d(kernel_size=down_fac, stride=down_fac))
            channels *= 2
        
        # Upsample blocks
        self.up_convs = nn.ModuleList()
        self.upsample = nn.ModuleList()
        # channels: 96 -> 48
        channels //= 2
        for idx in range(depth):
            # default channel: 96 -> 48 -> 48 -> 24
            # inter channels:        12 -> 12 -> 6
            if idx == 0:
                x_chns = channels
                y_chns = channels
                inter_channel = channels // 4
                out_chns = (x_chns + y_chns) // 2
            elif idx == 1:
                x_chns = channels
                y_chns = channels // 2
                inter_channel = channels // 4
                out_chns = (x_chns + y_chns) // 3
            else:
                channels //= 2
                x_chns = channels
                y_chns = channels // 2
                inter_channel = channels // 4
                out_chns = (x_chns + y_chns) // 3
            self.upsample.insert(
                0, UpsampleFusion(x_chns, y_chns, inter_channel, down_fac)
            )
            # input channels:  96 (48 + 48) -> 72 (48 + 24) -> 36 (24 + 12)
            # output channels: 48           -> 24           -> 12
            self.up_convs.insert(
                0, nn.Sequential(
                    nn.Conv1d(x_chns+y_chns, out_chns, 3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Conv1d(out_chns, out_chns, 3, padding=1),
                    nn.ReLU()
                )
            )
        
        self.final_conv = nn.Conv1d(init_channels, 1, 1)
        
    def forward(self, spec_input, time_input):
        # spec_input: (batch, frames, bins, 2)
        time_attn, freq_attn = self.double_attn(spec_input)
        x = self.dense(spec_input) + time_attn + freq_attn
        x = self.avg_pool(x)
        
        # Down path
        skips = []
        # (batch, bins, channels) -> (batch, channels, bins)
        x = x.permute(0, 2, 1)
        for i in range(self.depth):
            x = self.down_convs[i](x)
            skips.append(x)
            x = self.pools[i](x)
        
        # Time backbone (placeholder implementation)
        if self.use_time_backbone:
            weight_branch, feat_branch = self.time_backbone(time_input)   
            weight_branch = weight_branch.unsqueeze(2)  # (batch, ch, 1)
            feat_branch = feat_branch.unsqueeze(2)      # (batch, ch, 1)
            
            # 沿序列维度拼接
            weight_branch = torch.cat([weight_branch, x], dim=2)  # (batch, channels, seq_len+1)
            feat_branch = torch.cat([feat_branch, x], dim=2)      # (batch, channels, seq_len+1)

            weight_branch = self.weight_conv(weight_branch)
            feat_branch = self.feat_conv(feat_branch)

            # 特征融合
            x = x + weight_branch * feat_branch
        
        # Up path
        for i in reversed(range(self.depth)):
            x = self.upsample[i](x, skips[i])
            x = self.up_convs[i](x)
        
        x = self.final_conv(x)
        return x.view(x.size(0), -1)


# 完整模型使用方式保持不变
class BeliefPPG(nn.Module):
    def __init__(
            self,
            depth: int = 3,
            attn_channels: int = 32,
            init_channels: int = 12,
            down_fac: int = 4,
            n_frames: int = 7,
            n_bins: int = 64,
            freq: int = 64,
            min_hz: float = 0.5,
            max_hz: float =3.5,
            use_time_backbone=True,
            transition_prior=None,
            **kwargs
        ):
        super().__init__()
        self.hybrid_unet = HybridUNet(
            depth=depth,
            attn_channels=attn_channels,
            init_channels=init_channels,
            down_fac=down_fac,
            n_frames=n_frames,
            n_bins=n_bins,
            use_time_backbone=use_time_backbone
        )

        self.prior_layer = PriorLayer(
            dim=n_bins,
            min_hz=min_hz,
            max_hz=max_hz,
            online=True,
            return_probs=False,
            transition_prior=transition_prior,
            uncert="entropy"
        )
        
    def forward(self, spec_input, time_input, use_prior=False, need_logits_use_prior=False):
        logits = self.hybrid_unet(spec_input, time_input)
        logits = F.softmax(logits, dim=-1)
        if use_prior:
            if need_logits_use_prior:
                return logits, self.prior_layer(logits)
            return self.prior_layer(logits)
        return logits
    
    def online(self, online: bool = False):
        """设置在线模式"""
        self.prior_layer.online = online

    def fit_prior_layer(self, seq_data, distr_type = "gauss", sparse=False):
        """设置是否训练先验层"""
        self.prior_layer.fit_prior_layer(seq_data, distr_type, sparse)


if __name__ == "__main__":
    # 验证维度
    n_frames = 7
    n_bins = 64
    freq = 64
    attn_channels=32
    init_channels=12

    model = HybridUNet(
        attn_channels=attn_channels,
        init_channels=init_channels,
        down_fac=4,
        n_frames=n_frames,
        n_bins=n_bins,
        use_time_backbone=True
    )
    # model = BeliefPPG(n_frames, n_bins, freq, use_time_backbone=True)
    # (batch, frames, bins, 2)
    spec_input = torch.randn(2, n_frames, n_bins, 2)
    # (batch, seq_len, 1)
    # (freq * (n_frames - 1) * InputConfig.STRIDE + freq * InputConfig.WINSIZE, 1)
    # WINSIZE: 8, STRIDE: 2
    winsize = 8
    stride = 2
    time_input = torch.randn(2, freq * (n_frames - 1) * stride + freq * winsize, 1)
    output = model(spec_input, time_input)
    print("Output shape:", output.shape)  # 应该得到 (batch, n_bins)
