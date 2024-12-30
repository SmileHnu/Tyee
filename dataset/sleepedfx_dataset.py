#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : sleepedfx_dataset.py
@Time    : 2024/12/25 15:36:19
@Desc    : 
"""


import os
import torch
from torch.utils.data import Dataset
from dataset.base_dataset import BaseDataset, DatasetType

class SleepEDFxDataset(BaseDataset):
    def __init__(self, root, set_ids, transform=None):
        super().__init__(split=None)
        self.root = root
        self.set_ids = set_ids
        self.transform = transform
        self.data_files = self._load_data_files()

    def _load_data_files(self):
        data_files = []
        for root, dirs, files in os.walk(self.root):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                for sub_root, _, sub_files in os.walk(dir_path):
                    for set_id in self.set_ids:
                        matched_files = [os.path.join(sub_root, f) for f in sub_files if f.endswith(f".s{set_id}")]
                        data_files += matched_files
        return data_files

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        try:
            x = torch.load(file_path, weights_only=True)
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {e}")

        # 检查路径结构并提取标签
        try:
            y = int(file_path.split(os.sep)[-2])
            if y not in [0, 1, 2, 3, 4]:
                raise ValueError(f"Invalid label {y} in file path {file_path}")
        except Exception as e:
            raise ValueError(f"Error extracting label from file path {file_path}: {e}")

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
            "target": collated_labels
        }