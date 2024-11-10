#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : eeg2msvec.py
@Time    : 2024/10/14 16:44:36
@Desc    : 
"""
import math
import torch
import logging
import numpy as np
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

from .modules.ema import EMAModuleConfig, EMAModule
from .modules.model import EEG2MSVecModel


logger = logging.getLogger()


# Construct the tuple to record the mask info and seed
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])
MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])


def get_annealed_rate(
        start: int,
        end: int, 
        curr_step: int,
        total_steps: int
    ) -> float:
    """
    用于调整当前更新次数的指数移动平均的衰减率
    :param int start: 指数移动平均起始衰减率
    :param int end: 指数移动平均终止衰减率
    :param int curr_step: 指数移动平均当前步
    :param int total_steps: 指数移动平均的总步数
    :return float: 指数移动平均的衰减率
    """
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@dataclass
class EEG2MSVecConfig(FairseqDataclass):
    ### 1. Module: CNN feature extractor
    feat_enc_layers: str                = field(default="[(64, 32, 5), (64, 5, 2), (64, 3, 2)]")
    feat_dropout_prob: float            = 0.0
    feat_enc_bias: bool                 = False
    # feature extrctor update fraction
    local_grad_mult: float              = 1.0

    # feature projector dropout prob
    in_dropout_prob: float              = 0.0

    ### 2. Module: Positional encoder
    num_pos_layers: int                 = 4
    conv_pos_groups: int                = 2

    ### Masking 
    clone_batch: int                    = 8
    mask_noise_std: float               = 0.01
    mask_prob_min: Optional[float]      = None
    mask_prob: float                    = 0.5
    mask_length: int                    = 5
    inverse_mask: bool                  = False
    mask_dropout: float                 = 0.0
    add_masks: bool                     = False
    encoder_zero_mask: bool             = True
    # the minimal ratio to reserve
    keep_masked_pct: float              = 0.0
    mask_channel_prob: float            = 0.0
    mask_channel_length: int            = 1

    ### 3. Module: HRFormer Encoder
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

    ### 4. Module: Decoder
    decoder_dim: int                    = 1024
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


    ### 5. Module: Target representation 
    layer_norm_target_layer: bool       = False
    batch_norm_target_layer: bool       = False
    instance_norm_target_layer: bool    = False
    instance_norm_targets: bool         = False
    layer_norm_targets: bool            = False

    ### 6. Module: EMA module
    ema_decay: float                    = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool                = True
    log_norms: bool                     = True
    skip_ema: bool                      = False
    ema_end_decay: float                = field(default=0.9999, metadata={"help": "final ema decay rate"})
    # when to finish annealing ema decay rate
    ema_anneal_end_step: int            = II("optimization.max_update")
    ema_encoder_only: bool              = field(default=True, metadata={"help": "whether to momentum update only the shared transformer encoder"})


    ### 7. Module: Loss & Log metric
    min_target_var: float               = field(default=0.1, metadata={"help": "stop training if target var falls below this"})
    min_pred_var: float                 = field(default=0.01, metadata={"help": "stop training if prediction var falls below this"},)
    d2v_loss: float                     = 1
    decoder_group: bool                 = False
    loss_beta: float                    = field(default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"})
    loss_scale: Optional[float]         = field(default=None, metadata={"help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"},)

    # seed: int = II("common.seed")
    mae_init_weights: bool              = False


@register_model("eeg2msvec", dataclass=EEG2MSVecConfig)
class EEG2MSVec(BaseFairseqModel):
    def __init__(self, cfg: EEG2MSVecConfig, skip_ema=False, task=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.skip_ema = skip_ema

        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale
        self.local_grad_mult = cfg.local_grad_mult

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


        ### Build Model.
        self.model = EEG2MSVecModel(
            self.cfg
        )
        ### Initialize the parameters by MAE 
        self._init_model()

        # Exponentially moving average block
        self.ema = None
        # Wheather the EMA strategy use for target model.
        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay)

        self._reset_parameters()
        self.num_updates = 0

    def _init_model(self):
        def _init_weights(m: nn.Module):
            try:
                from apex.normalization import FusedLayerNorm
                fn = FusedLayerNorm
            except:
                fn = nn.LayerNorm

            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
        if self.cfg.mae_init_weights:
            self.apply(_init_weights)
        else:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params
            self.apply(init_bert_params)
    
    def _reset_parameters(self):
        for pn, p in self.named_parameters():
            # for bias with shape == 1, disable the weight decay
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            # assign the decoder parameter to optim group `decoder` with different optim strategy
            if self.cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

    @torch.no_grad()
    def make_ema_teacher(
        self,
        ema_decay: float
    ):
        """
        Construct the teacher agent, it will automatic update the teacher model
        :param float ema_decay: the EMA hyper-parameters
        """
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,
            ema_fp32=True,
            log_norms=self.cfg.log_norms,
            add_missing_params=False,
        )

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    def make_target_model(self):
        logger.info("making target model")
        # Construct the DyMS2Vec
        model_copy = EEG2MSVec(
            self.cfg, skip_ema=True, task=self.task
        )
        # 只拷贝编码编码器部分
        if self.cfg.ema_encoder_only:
            # 编码器（只对HRFormer使用EMA）
            model_copy: nn.Module = model_copy.model.encoder
            for p_s, p_t in zip(self.model.encoder.parameters(), model_copy.model.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            # 整个模型 （对CNN，HRFormer使用EMA）
            for p_s, p_t in zip(self.model.parameters(), model_copy.model.parameters()):
                p_t.data.copy_(p_s.data)
            # 设置 decoder 部分不采用EMA
            model_copy.model.decoder = None
        model_copy.requires_grad_(False)
        return model_copy
    
    def set_num_updates(self, num_updates):
        """ 用于设置当前训练的更新次数, 从而设置教师模型的移动平均 """
        super().set_num_updates(num_updates)

        # EMA 不为空, 教师模型存在, 且满足以下条件之一，则会跳过 EMA 更新
        # 1. 处于第一次训练过程 (前一次更新轮次为 0, 且当前更新轮次大于 1), 第一次更新之前不应该执行 EMA 更新.
        # 2. 先前更新次数大于等于当前更新轮次.
        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.model.encoder if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: EEG2MSVecConfig, task=None):
        """Build a new model instance."""
        return cls(cfg, task=task, skip_ema=cfg.skip_ema)
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor = None,
        id: torch.Tensor = None,
        padding_mask: torch.BoolTensor = None,
        mask: bool = True,
        features_only: bool = False,
        force_remove_masked: bool = False,
        precomputed_mask: bool = None,
    ):

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        # forward encoder model ( CNN + HRFormerEncoder )
        extractor_out = self.model(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        

        if features_only:
            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": None,
                "mask": encoder_mask,
            }

        xs = []
        if self.model.decoder is not None:
            # 进行解码
            dx = self.model.make_decoder_input(x, encoder_mask)
            dx = self.model.decoder(*dx)
            # shape [B, L, C]
            xs.append(dx)

        assert len(xs) > 0

        y = self._make_targets(source, target, extractor_out["local_features"], padding_mask)

        if self.cfg.clone_batch > 1:
            y = y.repeat_interleave(self.cfg.clone_batch, 0)

        # 获取掩码那些位置对应的标签
        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()
        # y shape [B, C, L]
        # masked_b shape [B, L]
        y = y.transpose(1, 2)
        y = y[masked_b] # shape [clone_batch * num_masked, C]

        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
        else:
            xs = [x.reshape(-1, x.size(-1)) for x in xs]

        sample_size = masked.sum().long()

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        if self.cfg.d2v_loss > 0:
            for i, x in enumerate(xs):
                reg_loss = self.d2v_loss(x, y)
                n = f"regression_{i}" if len(xs) > 1 else f"regression"
                result["losses"][n] = reg_loss * self.cfg.d2v_loss

        with torch.no_grad():
            if encoder_mask is not None:
                # ids_restore shape: {B, C, L}
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(-1) / encoder_mask.ids_restore.size(-1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var_{i}" if len(xs) > 1 else f"pred_var"
                result[n] = self.compute_var(x.float())
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var'].item()} < {self.cfg.min_target_var}, exiting"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var'].item()} < {self.cfg.min_target_var}, exiting"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.BoolTensor = None,
    ):

        # forward encoder model ( CNN + HRFormerEncoder )
        extractor_out = self.model(
            source,
            padding_mask,
            mask=False,
            remove_masked=False,
            clone_batch=1,
            mask_seeds=None,
            precomputed_mask=None,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        
        return {
            "x": x,
            "padding_mask": masked_padding_mask,
            "layer_results": None,
            "mask": encoder_mask,
        }

    def _make_ema_same_to(self, x: torch.Tensor):
        # 用于检测 EMA 的 target model 与输入是否具有相同的 type 与设备
        p: torch.Tensor = next(self.ema.model.parameters())
        # raw input x device and dtype
        device, dtype = x.device, x.dtype
        # the EMA model device and dtype
        ema_device, ema_dtype = p.device, p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        return ema_device, ema_dtype

    def _make_targets(self, x, target, local_features, padding_mask):
        _, ema_dtype = self._make_ema_same_to(x)
        # 使用 target model 获取表征标签
        tm = self.ema.model
        with torch.no_grad():
            tm.eval()
            # 若 EMA 仅用于 HRFormerEncoder 编码器，没有 CNN 等相关组件，则需要单独处理
            if self.cfg.ema_encoder_only:
                assert target is None
                # 获取 CNN 输出未掩码的表征
                # 调用 EMA 的 HRFormerEncoder 进行表征标签获取
                target_res = tm.contextualized_features(
                    local_features.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    force_remove_masked=False,
                )
            else:
                inp = (
                    target.to(dtype=ema_dtype)
                    if target is not None
                    else x.to(dtype=ema_dtype)
                )
                target_res = tm(
                    inp,
                    padding_mask=padding_mask,
                    features_only=True,
                    mask=False,
                    force_remove_masked=False,
                )
            y = target_res["x"]
        return self._postprocess_targets(y)

    def _postprocess_targets(
        self,
        y: torch.Tensor,
        branch_y: list[torch.Tensor] = None
    ):
        with torch.no_grad():
            target = y
            # BN, accept shape [B, C, L]
            if self.cfg.batch_norm_target_layer:
                target = F.batch_norm(
                    target.float(), 
                    running_mean=None, 
                    running_var=None, 
                    training=True
                )
            # IN, accept shape [B, C, L]
            if self.cfg.instance_norm_target_layer:
                target = F.instance_norm(target.float())
            # LN, accept shape [B, L, C]
            if self.cfg.layer_norm_target_layer:
                target = target.transpose(1, 2)
                target = F.layer_norm(target.float(), target.shape[-1:])
                target = target.transpose(1, 2)

        y = target.float()

        if self.cfg.layer_norm_targets:
            y = y.transpose(1, 2)
            y = F.layer_norm(y, y.shape[-1:])
            y = y.transpose(1, 2)

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y)
        return y
    
    def d2v_loss(self, x, y):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1))

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()
