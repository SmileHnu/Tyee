#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : kaggleern_dataset.py
@Time    : 2024/12/17 19:48:38
@Desc    : 
"""

import os
import csv
import torch
import pandas as pd
import numpy as np
import mne
import tqdm
from torch.utils.data import Dataset
from dataset import BaseDataset
mne.set_log_level("ERROR")
use_channels_names=[
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'P7', 'P3', 'PZ', 'P4', 'P8',
                'O1', 'O2'
    ]

# -- read Kaggle ERN
ch_names_kaggle_ern = list("Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(','))

class KaggleERNDataset(BaseDataset):
    def __init__(self, path, train, subjects=None, sessions=None, tmin=-0.7, tlen=2, data_max=None, data_min=None, use_channels_names=use_channels_names):
        self.path = path
        self.subjects = subjects if subjects is not None else [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26] if train else [1,3,4,5,8,9,10,15,19,25]
        self.sessions = sessions if sessions is not None else [1,2,3,4,5]
        self.tmin = tmin
        self.tlen = tlen
        self.data_max = data_max
        self.data_min = data_min
        self.use_channels_names = use_channels_names
        self.train = train
        self.labels = self._read_labels()
        self.data = self._read_data()

    def _read_labels(self):
        if self.train:
            labels = []
            with open(os.path.join(self.path, 'KaggleERN', 'TrainLabels.csv'), 'r') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i > 0:
                        labels.append(row)
            return dict(labels)
        else:
            labels = pd.read_csv(os.path.join(self.path, 'KaggleERN', 'true_labels.csv'))['label']
            return labels

    def _read_data(self):
        datas = []
        label_id = 0
        for i in tqdm.tqdm(self.subjects):
            for j in self.sessions:
                if i > 9:
                    if i == 22 and j == 5:
                        print("Skipped error file " + "KagglERN/train/Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
                        continue
                    filename = os.path.join(self.path, "KaggleERN", "train" if self.train else "test", "Data_S" + str(i) + "_Sess0" + str(j) + ".csv")
                else:
                    filename = os.path.join(self.path, "KaggleERN", "train" if self.train else "test", "Data_S0" + str(i) + "_Sess0" + str(j) + ".csv")

                for fb, trial in enumerate(read_csv_epochs(filename, tmin=self.tmin, tlen=self.tlen, data_max=self.data_max, data_min=self.data_min, use_channels_names=self.use_channels_names), 1):
                    if self.train:
                        label = self.labels["S{:02d}_Sess{:02d}_FB{:03d}".format(i, j, fb)]
                    else:
                        label = self.labels[label_id]
                        label_id += 1
                    datas.append((trial, int(label)))
        return datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label
    
    def collate_fn(self,batch):
        inputs, labels = zip(*batch)

        # 将输入序列按维度拼接
        collated_data = torch.stack(inputs, dim=0)
        return {
            "x": collated_data,
            "label": torch.tensor(labels).long()
        }

def read_csv_epochs(filename, tmin, tlen, use_channels_names=use_channels_names, data_max=None, data_min=None):
    sample_rate = 200
    raw = pd.read_csv(filename)
    
    data = torch.tensor(raw.iloc[:,1:-2].values) # exclude time EOG Feedback
    feed = torch.tensor(raw['FeedBackEvent'].values)
    stim_pos = torch.nonzero(feed>0)
    datas = []
    
    # -- get channel id by use chan names
    if use_channels_names is not None:
        choice_channels = []
        for ch in use_channels_names:
            choice_channels.append([x.lower().strip('.') for x in ch_names_kaggle_ern].index(ch.lower()))
        use_channels = choice_channels
    if data_max is not None: use_channels+=[-1]
    
    xform = lambda x: min_max_normalize(x, data_max, data_min)
    
    for fb, pos in enumerate(stim_pos, 1):
        start_i = max(pos + int(sample_rate * tmin), 0)
        end___i = min(start_i + int(sample_rate * tlen), len(feed))
        trial = data[start_i:end___i, :].clone().detach().cpu().numpy().T
        info = mne.create_info(
            ch_names=[str(i) for i in range(trial.shape[0])],
            ch_types="eeg",  # channel type
            sfreq=200,  # frequency
        )
        raw = mne.io.RawArray(trial, info)  # create raw
        trial = torch.tensor(raw.get_data()).float()
        trial = xform(trial)
        if use_channels_names is not None:
            trial = trial[use_channels]
        datas.append(trial)
    return datas

def min_max_normalize(x: torch.Tensor, data_max=None, data_min=None, low=-1, high=1):
    if data_max is not None:
        max_scale = data_max - data_min
        scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
        
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)
    x -= 0.5
    x += (high + low) / 2
    x  = (high - low) * x
    
    if data_max is not None:
        x = torch.cat([x, torch.ones((1, x.shape[-1])).to(x)*scale])
    return x
