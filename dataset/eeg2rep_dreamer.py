#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : eeg2rep_dreamer.py
@Time    : 2024/11/11 16:32:31
@Desc    : 
"""
import torch
import random
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from scipy.signal import resample_poly
from dataset.base_dataset import BaseDataset


class EEG2RepDREAMERDataset(BaseDataset):
    def __init__(
            self,
            source: Path,
            clip_length: int = 2,
            split: str = "train",
            pad: bool = False,
            drop: bool = True
        ) -> None:
        super().__init__()
        random.seed(575)
        self.pad = pad
        self.drop = drop
        self.random_crop = True
        self.max_sample_size = 1e+12

        if not source.exists():
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        
        self.data, self.targets = self._load_data(source, clip_length=clip_length, split=split)
    
    def _load_data(self, source: str, clip_length: int = 2, split: str = "train"):
        raw = np.load(source, allow_pickle=True)
        if split == "train":
            data = raw.item()["train_data"]
            label = raw.item()["train_label"]
        elif split == "dev":
            data = raw.item()["dev_data"]
            label = raw.item()["dev_label"]
        else:
            data = raw.item()["test_data"]
            label = raw.item()["test_label"]
        return data, label
        
    def __getitem__(self, idx):
        # shape: [L, C]
        # upsampled_signal = resample_poly(self.data[idx], up=(512 // 128), down=1)
        # data = torch.tensor(upsampled_signal * 1e-3, dtype=torch.float32)
        data = torch.tensor(self.data[idx] * 1e-3, dtype=torch.float32)
        data = torch.tensor(self.data[idx] * 1e-3, dtype=torch.float32)
        # print(data.shape)
        # shape: [C, L]
        # data = data.transpose(0, 1)
        # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        # data = data[2].unsqueeze(0)
        # with torch.no_grad():
            # data = F.layer_norm(data, normalized_shape=data.shape)

        # data = torch.tensor(self.data[idx], dtype=torch.float32)
        # data = data[2].unsqueeze(0)
        # return data, torch.tensor(self.targets[idx]).long()
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        data = data[2].unsqueeze(0)
        return data, torch.tensor(self.targets[idx]).long()

    def __len__(self):
        return len(self.targets)

    def collate_fn(
            self,
            samples: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        wavs, labels = [], []

        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        
        sizes = [w.shape[-1] for w in wavs]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        B, chn = len(wavs), wavs[0].shape[0]

        # print(target_size)
        collated_data = wavs[0].new_zeros(B, chn, target_size)
        padding_mask = (
            torch.BoolTensor(collated_data.shape).fill_(False) if self.pad else None
        )
        for i, data in enumerate(wavs):
            chn, num_frames = data.shape
            # print(chn, num_frames)
            diff = num_frames - target_size
            if diff == 0:
                # 长度等于 target_size，直接复制
                collated_data[i] = data
            elif diff < 0:
                # 长度小于 target_size，需要填充
                assert self.pad
                collated_data[i] = torch.cat(
                    [data, data.new_full((chn, -diff), 0.0)], dim=-1
                )
                padding_mask[i, :, diff:] = True
            else:
                # 长度大于 target_size，需要裁剪
                collated_data[i], _ = self.crop_to_max_size(data, target_size)
        return {
            "x": collated_data,
            "label": torch.tensor(labels).long(),
            "padding_mask": padding_mask
        }

    def crop_to_max_size(self, raw: torch.Tensor, target_size: int) -> tuple[torch.Tensor, int]:
        """
        crop the raw physio to the label size if the raw physio size is greater than label size
        :param torch.Tensor raw: the raw bio signal which is needed to crop
        :param int target_size: the crop size
        :return tuple[torch.Tensor, int]: the cropped bio signal and crop start position
        """
        assert raw.dim() == 2, "crop_to_max_size only support 2-dim data"
        _, size = raw.shape
        diff = size - target_size
        # if the data physio size if lower than label size, skip the crop ops
        if diff <= 0:
            return raw, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return raw[:, start:end], start


if __name__ == "__main__":
    # source = "data\DREAMER.mat"
    source = "/home/taoz/data/PhysioSignal/dreamer/DREAMER.mat"
    dataset = DREAMERDataset(Path(source))
    data = dataset[0]
    pass
