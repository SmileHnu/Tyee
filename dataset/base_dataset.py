#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : dataset.py
@Time    : 2024/09/25 17:02:30
@Desc    : 
"""
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels
    
    def build_transforms(self, cfg):
        pass