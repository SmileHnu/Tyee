#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv2a_dataset.py
@Time    : 2024/12/19 19:40:20
@Desc    : 
"""
import os
import torch
import scipy
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataset.base_dataset import BaseDataset, DatasetType

class BCICIV2ADataset(BaseDataset):
    def __init__(self, sub, data_path, few_shot_number=1, is_few_EA=False, target_sample=-1, use_avg=True, use_channels=None, split=DatasetType.UNKNOWN):
        super().__init__(split=split)
        self.sub = sub
        self.data_path = data_path
        self.few_shot_number = few_shot_number
        self.is_few_EA = is_few_EA
        self.target_sample = target_sample
        self.use_avg = use_avg
        self.use_channels = use_channels
        self.data, self.labels, self.subject_ids = self._load_data()

    def _load_data(self):
        target_session_1_path = os.path.join(self.data_path, f'sub{self.sub}_train/Data.mat')
        target_session_2_path = os.path.join(self.data_path, f'sub{self.sub}_test/Data.mat')

        session_1_data = sio.loadmat(target_session_1_path)
        session_2_data = sio.loadmat(target_session_2_path)
        R = None
        if self.is_few_EA:
            session_1_x = EA(session_1_data['x_data'], R)
        else:
            session_1_x = session_1_data['x_data']

        if self.is_few_EA:
            session_2_x = EA(session_2_data['x_data'], R)
        else:
            session_2_x = session_2_data['x_data']

        test_x_1 = torch.FloatTensor(session_1_x)
        test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

        test_x_2 = torch.FloatTensor(session_2_x)
        test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)

        if self.target_sample > 0:
            test_x_1 = temporal_interpolation(test_x_1, self.target_sample, use_avg=self.use_avg)
            test_x_2 = temporal_interpolation(test_x_2, self.target_sample, use_avg=self.use_avg)

        if self.use_channels is not None:
            test_input = torch.cat([test_x_1, test_x_2], dim=0)[:, self.use_channels, :]
        else:
            test_input = torch.cat([test_x_1, test_x_2], dim=0)

        test_labels = torch.cat([test_y_1, test_y_2], dim=0)

        source_train_x = []
        source_train_y = []
        source_train_s = []

        source_valid_x = []
        source_valid_y = []
        source_valid_s = []
        subject_id = 0
        for i in range(1, 10):
            if i == self.sub:
                continue
            train_path = os.path.join(self.data_path, f'sub{i}_train/Data.mat')
            train_data = sio.loadmat(train_path)

            test_path = os.path.join(self.data_path, f'sub{i}_test/Data.mat')
            test_data = sio.loadmat(test_path)
            if self.is_few_EA:
                session_1_x = EA(train_data['x_data'], R)
            else:
                session_1_x = train_data['x_data']

            session_1_y = train_data['y_data'].reshape(-1)

            train_x, valid_x, train_y, valid_y = train_test_split(session_1_x, session_1_y, test_size=0.1, stratify=session_1_y)

            source_train_x.extend(train_x)
            source_train_y.extend(train_y)
            source_train_s.append(torch.ones((len(train_y),)) * subject_id)

            source_valid_x.extend(valid_x)
            source_valid_y.extend(valid_y)
            source_valid_s.append(torch.ones((len(valid_y),)) * subject_id)

            if self.is_few_EA:
                session_2_x = EA(test_data['x_data'], R)
            else:
                session_2_x = test_data['x_data']

            session_2_y = test_data['y_data'].reshape(-1)

            train_x, valid_x, train_y, valid_y = train_test_split(session_2_x, session_2_y, test_size=0.1, stratify=session_2_y)

            source_train_x.extend(train_x)
            source_train_y.extend(train_y)
            source_train_s.append(torch.ones((len(train_y),)) * subject_id)

            source_valid_x.extend(valid_x)
            source_valid_y.extend(valid_y)
            source_valid_s.append(torch.ones((len(valid_y),)) * subject_id)
            subject_id += 1

        source_train_x = torch.FloatTensor(np.array(source_train_x))
        source_train_y = torch.LongTensor(np.array(source_train_y))
        source_train_s = torch.cat(source_train_s, dim=0)

        source_valid_x = torch.FloatTensor(np.array(source_valid_x))
        source_valid_y = torch.LongTensor(np.array(source_valid_y))
        source_valid_s = torch.cat(source_valid_s, dim=0)

        if self.target_sample > 0:
            source_train_x = temporal_interpolation(source_train_x, self.target_sample, use_avg=self.use_avg)
            source_valid_x = temporal_interpolation(source_valid_x, self.target_sample, use_avg=self.use_avg)

        if self.use_channels is not None:
            train_data = source_train_x[:, self.use_channels, :]
            valid_data = source_valid_x[:, self.use_channels, :]
        else:
            train_data = source_train_x
            valid_data = source_valid_x

        train_labels = source_train_y
        valid_labels = source_valid_y

        return (train_data, valid_data, test_input), (train_labels, valid_labels, test_labels), (source_train_s, source_valid_s)

    def __getitem__(self, idx):
        if self.split == DatasetType.TRAIN:
            data, labels, subject_ids = self.data[0], self.labels[0], self.subject_ids[0]
        elif self.split == DatasetType.DEV:
            data, labels, subject_ids = self.data[1], self.labels[1], self.subject_ids[1]
        elif self.split == DatasetType.TEST:
            data, labels = self.data[2], self.labels[2]
        else:
            # 报错
            raise ValueError("Invalid dataset split")
        return data[idx], labels[idx]

    def __len__(self):
        if self.split == DatasetType.TRAIN:
            return len(self.labels[0])
        elif self.split == DatasetType.DEV:
            return len(self.labels[1])
        elif self.split == DatasetType.TEST:
            return len(self.labels[2])
        else:
            # 报错
            raise ValueError("Invalid dataset split")
    
    def collate_fn(self, batch):
        inputs, labels = zip(*batch)

        # 将输入序列按维度拼接
        collated_data = torch.stack(inputs, dim=0)
        collated_labels = torch.stack(labels).long()
        
        return {
            "x": collated_data,
            "target": collated_labels,
            
        }


def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")


# 欧氏空间的对齐方式 其中x：NxCxS
def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x