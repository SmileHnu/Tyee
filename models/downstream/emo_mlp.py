#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : emo_mlp.py
@Time    : 2024/11/10 21:57:26
@Desc    : 
"""
from torch import nn
from torch.nn import functional as F


class EmoMLP(nn.Module):
    def __init__(self, num_classes: int = 2, num_features: int = 512):
        super(EmoMLP, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, *args, **kwargs):
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        return self.fc(x)
    
