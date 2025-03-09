#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mobi_dataset.py
@Time    : 2024/11/26 19:50:44
@Desc    : 
"""

import os
import torch
import pickle
from scipy.signal import resample

class MoBIDataset(torch.utils.data.Dataset):
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
        Y = sample["y"]
        X = torch.FloatTensor(X.values).transpose(0, 1)  # 将 DataFrame 转换为 numpy 数组并转置
        Y = torch.FloatTensor(Y.values)  # 将 Series 转换为 Tensor
        return X, Y
    
    def collate_fn(self, batch):
        inputs, labels = zip(*batch)

        # 将输入序列按维度拼接
        collated_data = torch.stack(inputs, dim=0)
        collated_labels = torch.stack(labels, dim=0)
        return {
            "x": collated_data,
            "label": collated_labels
        }
