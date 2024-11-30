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
import numpy as np
import pandas as pd
from glob import glob
from enum import Enum
from typing import Callable
from scipy.signal import resample
from dataset.base_dataset import BaseDataset

import pandas as pd
import torch
from enum import Enum
from typing import Callable

class DatasetType(Enum):
    UNKNOWN = "unknown"
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

class MoBIDataset(BaseDataset):
    def __init__(
        self,
        mobi_path: str = '',
        split: DatasetType = DatasetType.UNKNOWN,
        fs: int = 100,  # 当前采样率为100Hz
        target_fs: int = 200,  # 目标采样率为200Hz
        train_duration: int = 10,  # 训练集的时间长度（分钟）
        val_duration: int = 5,  # 验证集的时间长度（分钟）
        transform: Callable = None,
    ) -> None:
        super().__init__(split=split)
        self.fs = fs
        self.target_fs = target_fs  # 新增目标采样率
        self.mobi_path = mobi_path
        self.train_duration = train_duration
        self.val_duration = val_duration
        self.transform = transform

        # 加载并合并所有受试者的数据
        self.eeg_data, self.joints_data = self.load_and_merge_data()

    def load_and_merge_data(self):
        all_eeg_data = []
        all_joints_data = []

        folder_paths = glob(os.path.join(self.mobi_path, 'SL*-T*'))  # 获取 MoBI 文件夹中所有的 SLxx-Tyy 子文件夹

        for folder in folder_paths:
            eeg_file = os.path.join(folder, 'eeg.txt')  # EEG 数据文件路径
            joints_file = os.path.join(folder, 'joints.txt')  # 关节角度数据文件路径
            
            eeg_data, joints_data = self.load_data_from_files(eeg_file, joints_file)
            
            all_eeg_data.append(eeg_data)
            all_joints_data.append(joints_data)

        eeg_data = pd.concat(all_eeg_data, axis=0, ignore_index=True)
        joints_data = pd.concat(all_joints_data, axis=0, ignore_index=True)

        return eeg_data, joints_data

    def load_data_from_files(self, eeg_file, joints_file):
        eeg_data = pd.read_csv(eeg_file, delimiter="\t", header=None, skiprows=1)
        eeg_data = eeg_data.iloc[:, :-5]  # 去掉无用的列

        joints_data = pd.read_csv(joints_file, delimiter="\t", header=None, skiprows=2)
        joints_data = joints_data.iloc[:, :-1]  # 去掉时间戳列

        timestamps = eeg_data.iloc[:, 0].values
        if self.split.value == DatasetType.TRAIN.value:
            start_time = 2 * 60
            end_time = (self.train_duration + 2) * 60
        elif self.split.value == DatasetType.DEV.value:
            start_time = (self.train_duration + 2) * 60
            end_time = (self.val_duration + self.train_duration + 2) * 60
        elif self.split.value == DatasetType.TEST.value:
            start_time = 17 * 60
            end_time = 22 * 60
        else:
            raise ValueError(f"Unsupported split type: {self.split}")
        
        idx = (timestamps >= start_time) & (timestamps <= end_time)
        eeg_data = eeg_data[idx]
        joints_data = joints_data[idx]

        joints_data = joints_data.iloc[:, 1:]
        eeg_data = eeg_data.iloc[:, 1:]

        # 对EEG数据进行重采样
        eeg_data_resampled = self.resample_data(eeg_data)
        joints_data_resampled = self.resample_data(joints_data)

        return eeg_data_resampled, joints_data_resampled

    def resample_data(self, data):
        """
        对数据进行重采样，调整采样频率。
        :param eeg_data: 原始数据
        :return: 重采样后的数据
        """
        num_samples = int(data.shape[0] * self.target_fs / self.fs)
        resampled_data = np.zeros((num_samples, data.shape[1]))  # 创建一个新的空数组用于存储重采样后的数据

        # 对每一列通道进行重采样
        for i in range(data.shape[1]):
            resampled_data[:, i] = resample(data.iloc[:, i], num_samples)

        return pd.DataFrame(resampled_data, columns=data.columns)

    def normalize_angles(self, joints_data):
        return joints_data / 90.0

    def __getitem__(self, idx):
        """
        获取指定索引的样本。
        :param idx: 索引值
        :return: 返回一个样本，包含EEG数据和关节角度数据
        """
        # 每个样本长度是 2000（即 10 秒数据），假设每个patch为200个采样点
        sample_length = 2000  # 每个样本是 10 秒数据，采样率200Hz时是2000个采样点
        patch_size = 200  # 每个patch的大小（200个采样点）
        
        # 计算样本的patch数量
        num_patches = sample_length // patch_size  # 每个样本有多少个patch
        
        # 获取EEG样本数据，假设每个样本是连续的 10 秒数据（2000个采样点）
        if (idx + 1) * sample_length <= len(self.eeg_data):  # 确保索引不会超出数据范围
            eeg_sample = self.eeg_data.iloc[idx * sample_length : (idx + 1) * sample_length].values
            joints_sample = self.joints_data.iloc[idx * sample_length : (idx + 1) * sample_length].values
        else:
            # 如果数据不足2000个点，跳过这个样本
            raise IndexError(f"Index {idx} exceeds data length or insufficient data for a full sample.")
        
        # 只获取关节角度数据的最后一个采样点
        joints_sample = joints_sample[-1, :]  # 获取最后一个时间步的关节角度数据
        
        # 将EEG数据拆分为多个patches，形状为 [num_patches, patch_size, num_electrodes]
        eeg_sample = eeg_sample.reshape(num_patches, patch_size, 60)  # 将2000个采样点拆分成10个patch，每个patch 200个采样点
        eeg_sample = torch.tensor(eeg_sample, dtype=torch.float32)  # 转换为Tensor

        eeg_sample = eeg_sample.permute(2, 0, 1)  # 调整形状为 [num_electrodes, num_patches, patch_size]

        # 对关节角度数据进行归一化处理
        joints_sample = self.normalize_angles(joints_sample)

        # 创建最终的样本字典
        sample = {
            'x': eeg_sample,  # 输入数据
            'target': torch.tensor(joints_sample, dtype=torch.float32)  # 目标数据
        }

        # 如果有transform操作，应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


    def __len__(self):
        return len(self.eeg_data) // 2000  # 每个样本2000个采样点

    def collate_fn(self, samples):
        """
        合并批次样本，返回字典形式的数据。
        :param samples: 批次样本
        :return: 返回包含x（输入数据）和target（标签）的字典
        """
        wavs, labels = [], []
        for sample in samples:
            if sample['x'].shape[0] == 0:  # 如果样本为空，跳过该样本
                continue
            wavs.append(sample['x'])
            labels.append(sample['target'])

        if len(wavs) == 0:  # 如果批次内没有有效的样本，返回空
            raise ValueError("No valid samples in the batch.")

        # 将wavs和labels拼接成一个批次
        wavs = torch.stack(wavs, dim=0)
        labels = torch.stack(labels, dim=0)

        return {'x': wavs, 'target': labels}



if __name__ == "__main__":
    # 假设你的实际数据文件路径
    mobi_path = "data\MoBI"
    
    
    # 创建 MoBIDataset 实例，选择训练集（DatasetType.TRAIN）
    train_dataset = MoBIDataset(
        mobi_path=mobi_path,
        split=DatasetType.TRAIN,
    )
    print(len(train_dataset))
    # 测试 __getitem__ 方法（获取第一个样本）
    sample = train_dataset[0]
    print("First sample:")
    print(f"EEG shape: {sample['x'].shape}, Target shape: {sample['target'].shape}")
    print(f"EEG sample (first 5 channels): {sample['x'][:5]}")
    print(f"Target sample (first 5 targets): {sample['target'][:5]}")