#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : physiop300_dataset.py
@Time    : 2024/12/19 20:24:29
@Desc    : 
"""

import os
import torch
from torch.utils.data import Dataset
from dataset.base_dataset import BaseDataset, DatasetType

class PhysioP300Dataset(BaseDataset):
    def __init__(self, root, subs, transform=None):
        super().__init__(split=None)
        self.root = root
        self.subs = subs
        self.transform = transform
        self.data_files = self._load_data_files()

    def _load_data_files(self):
        data_files = []
        for sub in self.subs:
            data_files += [os.path.join(self.root, '1', f) for f in os.listdir(os.path.join(self.root, '1')) if f.endswith(f".sub{sub}")]
            data_files += [os.path.join(self.root, '0', f) for f in os.listdir(os.path.join(self.root, '0')) if f.endswith(f".sub{sub}")]
        return data_files

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        x = torch.load(file_path, weights_only=True)
        y = 1 if '1' in file_path.split(os.sep)[-2] else 0
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data_files)

    def collate_fn(self, batch):
        inputs, labels = zip(*batch)
        collated_data = torch.stack(inputs, dim=0)
        collated_labels = torch.tensor(labels).long()
        return {
            "x": collated_data,
            "label": collated_labels
        }