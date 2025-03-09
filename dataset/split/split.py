#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : split.py
@Time    : 2025/03/05 19:06:04
@Desc    : 
"""


from typing import Dict, List, Tuple, Union
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
from dataset import BaseDataset
import pandas as pd

class DatasetSplitter:
    """
    数据集划分工具类，用于将数据集划分为训练集、验证集和测试集。

    参数:
        train_dataset (BaseDataset): 训练数据集。
        dev_dataset (BaseDataset, 可选): 验证数据集，默认为 None。
        test_dataset (BaseDataset, 可选): 测试数据集，默认为 None。
    """
    def __init__(self, 
                 train_dataset: BaseDataset, 
                 dev_dataset: BaseDataset = None, 
                 test_dataset: BaseDataset = None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

    def split(self, method: str = 'none', split_by: str = 'record_id', seed: int = None, **kwargs) -> Union[Tuple[BaseDataset, BaseDataset, BaseDataset], List[Tuple[BaseDataset, BaseDataset, BaseDataset]]]:
        """
        划分数据集的方法。

        参数:
            method (str): 划分数据集的方法。可选值为 'none'、'kfold'、'leave_out'。
            split_by (str): 划分数据集的粒度。可选值为 'record_id'、'clip_id'、'subject_id'、'session_id'、'trial_id'。
            seed (int, 可选): 随机种子，用于数据划分的可重复性。
            **kwargs: 其他参数。

        返回:
            Union[Tuple[BaseDataset, BaseDataset, BaseDataset], List[Tuple[BaseDataset, BaseDataset, BaseDataset]]]: 划分后的数据集。
        """
        if method == 'none':
            # 不需要划分数据集
            return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

        elif method == 'kfold':
            # 按交叉验证的方式划分数据集
            n_splits = kwargs.get('n_splits', 5)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            unique_ids = self.train_dataset.info[split_by].unique()
            splits = [(self._create_subset(unique_ids[train_idx], split_by), self._create_subset(unique_ids[val_idx], split_by), self.test_dataset)
                      for train_idx, val_idx in kf.split(unique_ids)]
            return splits

        elif method == 'train_test_split':
            # 代码整合
            # 按留出法划分数据集
            unique_ids = self.train_dataset.info[split_by].unique()
            if self.dev_dataset is None and self.test_dataset is None:
                test_size = kwargs.get('test_size', 0.2)
                dev_size = kwargs.get('dev_size', 0.2)
                train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
                train_ids, dev_ids = train_test_split(train_ids, test_size=dev_size, random_state=seed)
                train_subset = self._create_subset(train_ids, split_by)
                val_subset = self._create_subset(dev_ids, split_by)
                test_subset = self._create_subset(test_ids, split_by)
                return [(train_subset, val_subset, test_subset)]
            elif self.dev_dataset is None:
                dev_size = kwargs.get('dev_size', 0.2)
                train_ids, dev_ids = train_test_split(unique_ids, test_size=dev_size, random_state=seed)
                train_subset = self._create_subset(train_ids, split_by)
                val_subset = self._create_subset(dev_ids, split_by)
                return [(train_subset, val_subset, self.test_dataset)]
            elif self.test_dataset is None:
                test_size = kwargs.get('test_size', 0.2)
                train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
                train_subset = self._create_subset(train_ids, split_by)
                test_subset = self._create_subset(test_ids, split_by)
                return [(train_subset, self.dev_dataset, test_subset)]
            else:
                return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

        else:
            raise ValueError(f"Unsupported split method: {method}")

    def _create_subset(self, ids: List[int], split_by: str) -> Subset:
        """
        根据ids创建BaseDataset子集。

        参数:
            ids (List[int]): 样本的ID列表。
            split_by (str): 划分数据集的粒度。

        返回:
            Subset: 子集。
        """
        indices = self.train_dataset.info[self.train_dataset.info[split_by].isin(ids)].index.tolist()
        return Subset(self.train_dataset, indices)