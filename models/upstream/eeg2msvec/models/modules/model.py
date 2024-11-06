#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : base_model.py
@Time    : 2024/09/11 22:24:20
@Desc    : 
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional
from fairseq.data.data_utils import compute_mask_indices
from .hrformer import (
    TemporalFeatureExtractor,
    HRFormerEncoder
)
from models.modules.transformer import ConvPositionEmbedding
from models.modules.decoder import Decoder1d
from models.modules.utils import GradMultiply, Transpose


# Construct the tuple to record the mask info and seed
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])
MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])


@dataclass
class EEG2MSVecModelConfig:
    # CNN feature extractor configuration
    feat_enc_layers: str                = field(default="[(64, 32, 5), (64, 5, 2), (64, 3, 2)]")
    feat_dropout_prob: float            = 0.0
    feat_enc_bias: bool                 = False

    local_grad_mult: float              = 0.1
    in_dropout_prob: float                = 0.0

    # Positional encoder
    num_pos_layers: int                 = 6
    conv_pos_groups: int                = 6

    # HRFormer Encoder
    num_modules: str                    = field(default="[1, 1, 1, 1]")
    in_channels: str                    = field(default="[[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]")
    num_blocks: str                     = field(default="[[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]]")
    ffn_embed_dims: str                 = field(default="[[256], [256, 512], [256, 512, 1024], [256, 512, 1024, 2048]]")
    num_attn_heads: str                 = field(default="[[1], [1, 2], [1, 2, 4], [1, 2, 4, 8]]")
    dropout_probs: str                  = field(default="[[0.1], [0.1] * 2, [0.1] * 3, [0.1] * 4]")
    attn_dropout_probs: str             = field(default="[[0.1], [0.1] * 2, [0.1] * 3, [0.1] * 4]")
    acti_dropout_probs: str             = field(default="[[0.0], [0.0] * 2, [0.0] * 3, [0.0] * 4]")
    norm_first: str                     = field(default="[[False], [False] * 2, [False] * 3, [False] * 4]")
    fused_embed_dim: int                = 256
    align_corners: bool                 = False

    # Masking 
    mask_prob: float                    = 0.5
    mask_prob_min: Optional[float]      = None
    mask_length: int                    = 1
    inverse_mask: bool                  = True
    mask_dropout: float                 = 0.0
    add_masks: bool                     = False
    # the minimal ratio to reserve
    keep_masked_pct: float              = 0.2           
    encoder_zero_mask: bool             = True
    mask_noise_std: float               = 0.2
    mask_channel_prob: float            = 0.2
    mask_channel_length: int            = 1

    # Decoder
    decoder_dim: int                    = 256
    decoder_groups: int                 = 16
    decoder_kernel: int                 = 5
    decoder_layers: int                 = 5
    decoder_residual: bool              = True
    input_drop_prob:float               = 0.1
    proj_layers: int                    = 1
    proj_ratio: int                     = 2
    add_position_mask: bool             = False
    add_position_all: bool              = False
    # decoder to multiple branches
    multi_branch: bool                  = False


class EEG2MSVecModel(nn.Module):
    def __init__(
            self,
            cfg: EEG2MSVecModelConfig,
        ) -> None:
        super().__init__()

        self.cfg = cfg
        # CNN
        self.feat_enc_layers = eval(self.cfg.feat_enc_layers)
        # HRFormer
        self.num_modules = eval(self.cfg.num_modules)
        self.in_channels = eval(self.cfg.in_channels)
        self.num_blocks = eval(self.cfg.num_blocks)
        self.ffn_embed_dims = eval(self.cfg.ffn_embed_dims)
        self.num_attn_heads = eval(self.cfg.num_attn_heads)
        self.dropout_probs = eval(self.cfg.dropout_probs)
        self.attn_dropout_probs = eval(self.cfg.attn_dropout_probs)
        self.acti_dropout_probs = eval(self.cfg.acti_dropout_probs)
        self.norm_first = eval(self.cfg.norm_first)

        self.local_grad_mult = self.cfg.local_grad_mult

        # Masking 
        self.mask_prob = self.cfg.mask_prob
        self.mask_prob_min = self.cfg.mask_prob_min
        self.mask_length = self.cfg.mask_length
        self.inverse_mask = self.cfg.inverse_mask
        self.mask_dropout = self.cfg.mask_dropout
        self.add_masks = self.cfg.add_masks
        self.keep_masked_pct = self.cfg.keep_masked_pct
        self.encoder_zero_mask = self.cfg.encoder_zero_mask
        self.mask_noise_std = self.cfg.mask_noise_std
        self.mask_channel_prob = self.cfg.mask_channel_prob
        self.mask_channel_length = self.cfg.mask_channel_length

        self._make_encoder()
        self._make_decoder()

    def _make_encoder(self):
        ### module 1. CNN Feature Extractor
        # input shape: BxCxT
        self.local_encoder = TemporalFeatureExtractor(
            feat_enc_layers=self.feat_enc_layers,
            dropout=self.cfg.feat_dropout_prob,
            conv_bias=self.cfg.feat_enc_bias
        )

        ### module 2. Projector mapped token to HRFormerEncoder input channel
        embed_in_dim = self.feat_enc_layers[-1][0]
        embed_out_dim = self.in_channels[0][0]
        # input shape: BxCxT
        self.feature_projector = nn.Sequential(
            Transpose(dim0=1, dim1=2),
            nn.LayerNorm(embed_in_dim),
            nn.Linear(embed_in_dim, embed_out_dim),
            Transpose(dim0=1, dim1=2),
        )
        self.dropout_input = nn.Dropout(p=self.cfg.in_dropout_prob)

        ### module 3. Positional Encoder module 
        # NOTICE: the absolute positional encoder will be utilized before the Mask Ops.
        self.fixed_positional_encoder = None
        # NOTICE: the relative positional encoder will be utilized after the Mask Ops.
        # it accept input data shape: BxCxT
        self.relative_positional_encoder = ConvPositionEmbedding(
            conv_pos_dim=embed_out_dim, conv_pos_layers=self.cfg.num_pos_layers, conv_pos_groups=self.cfg.conv_pos_groups
        )

        ### module 4. Construct HRFormer Encoder, input shape: BxCxT
        self.encoder = HRFormerEncoder(
            num_modules=self.num_modules,
            in_channels=self.in_channels,
            num_blocks=self.num_blocks,
            ffn_embed_dims=self.ffn_embed_dims,
            num_attn_heads=self.num_attn_heads,
            dropout_probs=self.dropout_probs,
            attn_dropout_probs=self.attn_dropout_probs,
            acti_dropout_probs=self.acti_dropout_probs,
            norm_first=self.norm_first,
            align_corners=False
        )

    def _make_decoder(self):
        ### module 5. Construct Decoder 
        # dim = self.cfg.fused_embed_dim * len(self.in_channels)
        # assert dim == self.cfg.decoder_dim, "The fused_embed_dim * len(in_channels) should be equal to decoder_dim"
        self.decoder = Decoder1d(
            decoder_dim = self.cfg.fused_embed_dim,
            decoder_groups = 16,
            decoder_kernel = 5,
            decoder_layers = 5,
            decoder_residual = True,
            input_drop_prob = 0.1,
            proj_layers = 1,
            proj_ratio = 2,
            position_mask = False,
            position_all = False,
            input_dim = self.cfg.decoder_dim
        )

    def forward(
        self, 
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        mask: bool = True,
        remove_masked: bool = True,
        clone_batch: int = 1,
        mask_seeds: Optional[MaskSeed] = None,
        precomputed_mask: bool = False,
    ) -> torch.Tensor:
        return self.encode(x, padding_mask, mask, remove_masked, clone_batch, mask_seeds, precomputed_mask)
        
    def local_features(self, x) -> torch.Tensor:
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(x)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(x), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(x)

        x = self.feature_projector(x)
        return x
    
    def contextualized_features(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        mask: bool = True,
        remove_masked: bool = True,
        clone_batch: int = 1,
        mask_seeds: Optional[MaskSeed] = None,
        precomputed_mask: bool = False
    ) -> torch.Tensor:

        # the padding mask is computed on raw input data
        # it will downsampled when using CNN feature extractor
        if padding_mask is not None:
            # input shape: BxCxT
            padding_mask = self._convert_key_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        mask_info = None

        # Absolute positional encoding is added before masking to ensure 
        # that the model has access to the original position of each token, 
        # regardless of whether it is masked later.
        # input shape: BxCxL
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        # do mask strategy
        if mask:
            # wheather do multiple mask
            if clone_batch > 1:
                # clone the `x` according to the clone_batch
                # it will clone the `x` for clone_batch times on the dim=0 
                # if input x shape is [B, L, C] -> [B * clone_batch, L, C]
                # if input x shape is [B, C, L] -> [B * clone_batch, L, C]
                x = x.repeat_interleave(clone_batch, 0)
                # ensure the each time of clone has the different mask
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            # masking
            # need x shape: BxLxC
            x, mask_info = self._compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )
        
        # Relative positional encoding is applied after masking to ensure 
        # that the model focuses on the relative distances between visible (unmasked) tokens, 
        # preventing masked tokens from affecting the relative position information.
        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)
        

        masked_padding_mask = padding_mask
        # if mask strategy applied and remove the masked tokens
        # the positional encoding will gather based on the unmasked tokens
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x_pos = torch.gather(
                    x_pos, dim=2, index=mask_info.ids_keep
                )

            # padding mask
            if padding_mask is not None:
                # masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                # padding_mask shape: [B, L], while mask_info.ids_keep shape [B, C, L]
                masked_padding_mask = torch.gather(
                    padding_mask,
                    dim=1,
                    index=mask_info.ids_keep[:, 0, :],  # ignore the feature dimension
                )
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None
        
        if x_pos is not None:
            x = x + x_pos

        # input shape: BxCxL
        x, branch_x = self.encoder(
            x,
            key_padding_mask=masked_padding_mask,
            need_branches=True
        )

        return {
            "x": x,
            "branch_x": branch_x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "encoder_mask": mask_info
        }

    def make_decoder_input(self, x: torch.Tensor, mask_info: MaskInfo):
        # dropout input before decoder
        inp_drop = self.cfg.input_drop_prob
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        if mask_info is not None:
            # 计算被掩码的token数量：使用ids_restore中的shape信息， shape BxLxD
            # ids_restore是一个还原序列的索引，其记录了哪些位置用于还原，因此通过它计算mask掉的token数量
            # num_masked = mask_info.ids_restore.shape[1] - x.shape[1]
            num_masked = mask_info.ids_restore.shape[-1] - x.shape[-1]
            # 创建形状为(batch_size, num_masked, feature_dim)的掩码token张量
            # 这些mask token使用标准正态分布进行初始化，模仿随机噪声的效果
            # x shape [B, C, L]
            # noise shape: [B, L, C] -> normal -> [B, C, L]
            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(1),
            ).normal_(0, self.cfg.mask_noise_std).transpose(1, 2)

            # 将输入 x 的部分和 mask token 拼接到一起，形成完整的序列（包括未掩码和掩码的部分）
            x_ = torch.cat([x, mask_tokens], dim=2)
            # 使用mask_info中的ids_restore进行还原操作，将掩码token与真实token按原始顺序重组
            # ids_restore定义了如何将被掩码的token还原到原始位置
            ids_restore = mask_info.ids_restore[:, 0, :].unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.gather(x_, dim=2, index=ids_restore)

            # 如果配置中设置了 add_positions_masked 选项，将位置编码添加到还原的的 token 上 (添加随机生成的token并按照顺序重新排列)
            if self.cfg.add_position_mask:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                # mask_info.mask shape is [B, L]
                # pos shape is [B, C, L]
                x = x + (pos * mask_info.mask.unsqueeze(1))

         # 如果配置中设置了 add_positions_all 选项，为所有 token 添加位置编码
        if self.cfg.add_position_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        mask: bool = True,
        remove_masked: bool = True,
        clone_batch: int = 1,
        mask_seeds: Optional[MaskSeed] = None,
        precomputed_mask: bool = False,
    ):
        # use CNN to capture local information
        # input shape: BxCxT
        x = self.local_features(x)
        if self.dropout_input is not None:
            x = self.dropout_input(x)

        # use HRFormer to capture multiple-level information
        return self.contextualized_features(
            x,
            padding_mask=padding_mask,
            mask=mask,
            remove_masked=remove_masked,
            clone_batch=clone_batch,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

    def decode(self, x):
        pass

    def _convert_key_padding_mask(
        self,
        x: torch.Tensor, 
        padding_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, C, L = x.shape
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """
            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)
            for i in range(len(self.feat_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feat_enc_layers[i][1],
                    self.feat_enc_layers[i][2],
                )
            return input_lengths.to(torch.long)
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(size=(B, L), dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    size=(B, L), dtype=torch.bool, device=x.device
                )
        return padding_mask

    def _compute_mask(
        self, 
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        mask_seed: MaskSeed,
        apply: bool = False,
        precomputed_mask: bool = False
    ) -> tuple[torch.Tensor, MaskInfo]:
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self._make_maskinfo_from_mask(x, mask)
        else:
            # B, L, C = x.shape
            B, C, L = x.shape
            
            if (
                self.mask_prob_min is not None
                and self.mask_prob_min >= 0
                and self.mask_prob_min < self.mask_prob
            ):
                mask_prob = np.random.uniform(self.mask_prob_min, self.mask_prob)
            else:
                mask_prob = self.mask_prob

            if mask_prob:
                # random masking when the mask length == 1
                if self.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    # inverse block mask when the mask_length >= 1 and inverse_mask == True
                    if self.inverse_mask:
                        mask_prob = 1 - mask_prob

                    # got the mask indices
                    mask = compute_mask_indices(
                        (B, L),
                        padding_mask,
                        mask_prob,
                        self.mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=self.mask_dropout,
                        add_masks=self.add_masks,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )
                    mask = torch.from_numpy(mask).to(device=x.device)
                    if self.inverse_mask:
                        mask = 1 - mask
                    mask_info = self._make_maskinfo_from_mask(x, mask)
            else:
                mask_info = None

        # apply mask to data `x`
        if apply:
            x = self._apply_mask(x, mask_info)

        return x, mask_info

    def _make_maskinfo_from_mask(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor,
        shape: Optional[bool] = None
    ) -> MaskSeed:
        """
        got the MaskInfo(x_unmasked, mask, ids_restore, ids_keep) according to mask booled matrix
        :param torch.Tensor x: raw data x, shape [B, C, L]
        :param torch.BoolTensor mask: mask matrix, shape[B, L], `True` means mask the position value, `False` means reserve
        :return MaskSeed: 
        """
        if shape is None:
            B, C, L = x.shape
        else:
            B, C, L = shape
        
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        # sort and expand shape [B, L] -> [B, C, L]
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(dim=1).expand(-1, C, -1)
        # all sample mask same length
        len_keep = L - mask[0].sum()
        
        # it set min ratio to reserve 
        if self.keep_masked_pct > 0:
            len_keep += round((L - int(len_keep)) * self.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]
        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(dim=1).expand(-1, C, -1)
            x_unmasked = torch.gather(
                x, dim=2, index=ids_keep
            )
        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep
        )
        return mask_info

    def _apply_mask(
        self, 
        x: torch.Tensor, 
        mask_info: MaskInfo
    ) -> torch.Tensor:
        B, C, L = x.shape
        if mask_info is not None:
            # shape: [B, L]
            mask: torch.Tensor = mask_info.mask
            if self.encoder_zero_mask:
                # x shape [B, C, L], while mask shape [B, L]
                x = x * (1 - mask.type_as(x).unsqueeze(1))
            else:
                num_masks = mask.sum().item()
                # shape [NumOfMask, C]
                masks = x.new_empty(num_masks, C).normal_(0, self.mask_noise_std)
                # [B, L, C]
                x = x.permute(0, 2, 1)
                # it cause inplace problem when backward
                # x[mask] = masks
                x = index_put(x, mask, masks)
                # 恢复为 [B, C, L]
                x = x.permute(0, 2, 1)
            
        if self.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                shape=(B, C),
                padding_mask=None,
                mask_prob=self.mask_channel_prob,
                mask_length=self.mask_channel_length
            )
            mask_channel = torch.from_numpy(
                mask_channel
            ).to(x.device).unsqueeze(-1).expand(-1, -1, L)
            x[mask_channel] = 0
        return x


def random_masking(
        x: torch.Tensor,
        mask_ratio: float,
        mask_seed: Optional[MaskSeed]
    ):
    """
    random masking ops
    :param torch.Tensor x: input x need to mask, shape: [B, C, L]
    :param float mask_ratio: mask ratio
    :param Optional[MaskSeed] mask_seed: mask seed for re-production
    :return 
    """
    # batch, dim, length
    B, C, L = x.shape
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(B, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # reserve the len_keep tokens
    ids_keep = ids_shuffle[:, :len_keep]
    # ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, C)
    # ids_keep shape: [B, L] -> [B, 1, L] -> [B, C, L]
    ids_keep = ids_keep.unsqueeze(1).expand(-1, C, -1)
    # got the unmasked token
    x_unmasked = torch.gather(x, dim=2, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask = mask.to(torch.uint8)

    # ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, C)
    ids_restore = ids_restore.unsqueeze(1).expand(-1, C, -1)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def index_put(
        tensor: torch.Tensor,
        indices: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
    tensor[indices] = value
    return tensor


if __name__ == "__main__":
    # 创建 OmegaConf 配置
    from omegaconf import OmegaConf
    conf = OmegaConf.structured(DyMS2VecModelConfig)
    print(conf.num_modules)
    pass

