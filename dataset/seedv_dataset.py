#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : seedv_dataset.py
@Time    : 2024/12/26 20:12:55
@Desc    : 
"""
import os
import torch
import pickle
from scipy.signal import resample

class SEEDVDataset(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["y"])
        X = torch.FloatTensor(X)
        return X, Y
    
    def collate_fn(self, batch):
        inputs, labels = zip(*batch)

        # 将输入序列按维度拼接
        collated_data = torch.stack(inputs, dim=0)
        return {
            "x": collated_data,
            "label": torch.tensor(labels).long()
        }
