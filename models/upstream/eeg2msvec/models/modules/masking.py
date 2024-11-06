#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : masking.py
@Time    : 2024/09/12 20:35:25
@Desc    : 
"""
import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MaskingConfig:
    mask_prob: float = 0.5
    mask_prob_min: Optional[float] = None
    mask_length: int = 1
    inverse_mask: bool = True
    mask_dropout: float = 0.0
    add_masks: bool = False
    # the minimal reserved ratio
    keep_masked_pct: float = 0.2
    # use zero mask
    encoder_zero_mask: bool = True
    # use gussian mask
    mask_noise_std: float = 0.2
    mask_channel_prob: float = 0.2
    mask_channel_length: float = 1


@dataclass
class MaskSeed:
    seed: int = 0
    update: int = 0
    ids: int = 0


@dataclass
class MaskInfo:
    x_unmasked: torch.Tensor
    mask: torch.BoolTensor
    ids_restore: torch.LongTensor
    ids_keep: torch.LongTensor



class Masking:
    def __init__(
            self,
            cfg: MaskingConfig
        ) -> None:
        
        self.mask_prob = cfg.mask_prob
        self.mask_prob_min = cfg.mask_prob_min
        self.mask_length = cfg.mask_length
        self.inverse_mask = cfg.inverse_mask
        self.mask_dropout = cfg.mask_dropout
        self.add_masks = cfg.add_masks
        self.keep_masked_pct = cfg.keep_masked_pct
        self.encoder_zero_mask = cfg.encoder_zero_mask
        self.mask_noise_std = cfg.mask_noise_std
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_length = cfg.mask_channel_length


    def compute_mask(
            self, 
            x: torch.Tensor,
            padding_mask: Optional[torch.Tensor],
            mask_seed: MaskSeed,
            apply: bool = False,
            precomputed_mask: bool = False
        ) -> tuple[torch.Tensor, MaskInfo]:
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo_from_mask(x, mask)
        else:
            B, L, D = x.shape
            
            if (
                self.mask_prob_min is not None
                and self.mask_prob_min >= 0
                and self.mask_prob_min < self.mask_prob
            ):
                mask_prob = np.random.uniform(self.mask_prob_min, self.mask_prob)
            else:
                mask_prob = self.mask_prob

            if mask_prob > 0:
                # random masking when the mask length == 1
                if self.mask_length == 1:
                    mask_info = random_masking(
                        x, mask_ratio=mask_prob, mask_seed=None
                    )
                else:
                    # inverse block mask when the mask_length >= 1 and inverse_mask == True
                    if self.inverse_mask:
                        mask_prob = 1 - mask_prob

                    # got the mask matrix, shape [B, T], filled with True when masked
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
                    mask_info = self.make_maskinfo_from_mask(x, mask)
            else:
                mask_info = None

        # apply mask to data `x`
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info
    
    def make_maskinfo_from_mask(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor,
        shape: Optional[bool] = False
    ) -> MaskSeed:
        """
        got the MaskInfo(x_unmasked, mask, ids_restore, ids_keep) according to mask booled matrix
        :param torch.Tensor x: raw data x
        :param torch.BoolTensor mask: mask matrix, shape[B, T], `True` means mask the position value, `False` means reserve
        :return MaskSeed: 
        """
        if shape is None:
            B, L, D = x.shape
        else:
            B, L, D = shape
        
        torch.argsort()
        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        # sort and expand shape [B, T] -> [B, T, D]
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(dim=-1).expand(-1, -1, D)
        # all sample mask same length
        len_keep = L - mask[0].sum()
        
        # the minimal reserved ratio
        if self.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]
        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(dim=-1).expand(-1, -1, D)
            x_unmasked = torch.gather(
                x, dim=1, index=ids_keep
            )
        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep
        )
        return mask_info

    def apply_mask(
            self, 
            x: torch.Tensor, 
            mask_info: MaskInfo
        ) -> torch.Tensor:
        B, T, C = x.shape
        if mask_info is not None:
            mask: torch.BoolTensor = mask_info.mask
            if self.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(0, self.mask_noise_std)
                x[mask] = masks
            
        if self.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                shape=(B, C),
                padding_mask=None,
                mask_prob=self.mask_channel_prob,
                mask_length=self.mask_channel_length
            )
            mask_channel = torch.from_numpy(
                mask_channel
            ).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel] = 0
        return x


def random_masking(
        x: torch.Tensor,
        mask_ratio: float,
        mask_seed: Optional[MaskSeed]
    ):
    # batch, length, dim
    B, L, D = x.shape
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

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask