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

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit
)
from torch.utils.data import Subset
from dataset import BaseDataset


class DatasetSplitter:
    """
    数据集划分工具类，用于将生理信号数据集划分为训练集、验证集和测试集。

    参数:
        train_dataset (BaseDataset): 训练数据集（包含完整 info 数据）。
        dev_dataset (Optional[BaseDataset]): 验证数据集，默认为 None。
        test_dataset (Optional[BaseDataset]): 测试数据集，默认为 None。

    常见的划分方法包括：
        - none: 不进行划分，直接返回输入数据集。
        - kfold: 使用 K 折交叉验证，支持分层（适用于 clip_id 等）。
        - hold_out: 按留出法划分训练/验证/测试集。
        - loso: Leave-One-Subject-Out 划分（例如：对 subject_id 划分）。
        - loto: Leave-One-Trial-Out 划分（对 trial_id 划分）。
        - seedv: 针对 SEED-V 数据集的自定义划分（每个 session 内试验 5:5:5）。
        - ninapro: 针对 Ninapro 数据集的划分（基于 trial 和 stimulus）。
    """

    def __init__(
        self,
        train_dataset: BaseDataset,
        dev_dataset: Optional[BaseDataset] = None,
        test_dataset: Optional[BaseDataset] = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

    def split(
        self,
        method: str = "none",
        split_by: str = "record_id",
        stratified: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[BaseDataset, Optional[BaseDataset], Optional[BaseDataset]],
        List[Tuple[BaseDataset, Optional[BaseDataset], Optional[BaseDataset]]],
    ]:
        """
        根据指定的方法划分数据集。

        参数:
            method (str): 划分方法。可选值包括：
                'none', 'kfold', 'hold_out', 'loso', 'loto', 'seedv', 'ninapro'
            split_by (str): 划分的粒度，例如 'record_id', 'clip_id',
                            'subject_id', 'session_id', 'trial_id'。
            stratified (bool): 是否采用分层采样（例如对 clip_id）。
            seed (Optional[int]): 随机种子。
            shuffle (bool): 是否打乱顺序。
            **kwargs: 其他划分参数，如 n_splits, test_size, dev_size 等。

        返回:
            如果 method 为 'none'，返回单个三元组；
            否则返回包含多个划分结果的列表，每个元素为 (train, dev, test) 三元组。
        """
        if method == "none":
            return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

        elif method == "kfold":
            return self._split_kfold(split_by, stratified, seed, shuffle, **kwargs)

        elif method == "hold_out":
            return self._split_hold_out(split_by, stratified, seed, shuffle, **kwargs)

        elif method == "loso":
            # 留一被试/会话划分（如 subject_id 或 session_id）
            return self._split_leave_one(split_by)

        elif method == "loto":
            # 留一试验划分（以 trial_id 为粒度）
            return self._split_leave_one(split_by="trial_id")

        elif method == "seedv":
            return self._split_seedv(split_by, seed)

        elif method == "ninapro":
            return self._split_ninapro()

        else:
            raise ValueError(f"Unsupported split method: {method}")

    def _create_subset(self, ids: List, split_by: str) -> Subset:
        """
        根据给定的 ID 列表和分割依据，在训练集上创建子集。

        参数:
            ids (List): 要选择的唯一 ID 列表。
            split_by (str): 划分数据集的粒度。

        返回:
            Subset: 对应样本的子集。
        """
        indices = self.train_dataset.info[
            self.train_dataset.info[split_by].isin(ids)
        ].index.tolist()
        return Subset(self.train_dataset, indices)

    def _split_kfold(
        self,
        split_by: str,
        stratified: bool,
        seed: Optional[int],
        shuffle: bool,
        **kwargs,
    ) -> List[Tuple[BaseDataset, BaseDataset, Optional[BaseDataset]]]:
        n_splits = kwargs.get("n_splits", 5)
        if not shuffle:
            seed = None
        # 当粒度为 clip_id 且需要分层时使用 StratifiedKFold
        if split_by == "clip_id" and stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            labels = self.train_dataset.info["label"]
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
            labels = None

        unique_ids = self.train_dataset.info[split_by].unique()
        splits = []
        for train_idx, val_idx in kf.split(unique_ids, labels):
            train_ids = list(unique_ids[train_idx])
            val_ids = list(unique_ids[val_idx])
            train_subset = self._create_subset(train_ids, split_by)
            val_subset = self._create_subset(val_ids, split_by)
            splits.append((train_subset, val_subset, self.test_dataset))
        return splits

    def _split_hold_out(
        self,
        split_by: str,
        stratified: bool,
        seed: Optional[int],
        shuffle: bool,
        **kwargs,
    ) -> List[Tuple[BaseDataset, Optional[BaseDataset], Optional[BaseDataset]]]:
        unique_ids = self.train_dataset.info[split_by].unique()
        test_size = kwargs.get("test_size", 0.2)
        dev_size = kwargs.get("dev_size", 0.2)

        if stratified and split_by == "clip_id":
            labels = self.train_dataset.info["label"]
            split_fn = self._stratified_split
        else:
            labels = None
            split_fn = self._random_split

        if self.dev_dataset is None and self.test_dataset is None:
            train_ids, test_ids = split_fn(unique_ids, labels, test_size, seed, shuffle)
            # 在训练集上再次划分出验证集
            train_ids, dev_ids = split_fn(
                train_ids,
                labels[train_ids] if labels is not None else None,
                dev_size,
                seed,
                shuffle,
            )
        elif self.dev_dataset is None:
            train_ids, dev_ids = split_fn(unique_ids, labels, dev_size, seed, shuffle)
            test_ids = None
        elif self.test_dataset is None:
            train_ids, test_ids = split_fn(unique_ids, labels, test_size, seed, shuffle)
            dev_ids = None
        else:
            return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

        train_subset = self._create_subset(list(train_ids), split_by)
        dev_subset = self._create_subset(list(dev_ids), split_by) if dev_ids is not None else self.dev_dataset
        test_subset = (
            self._create_subset(list(test_ids), split_by)
            if test_ids is not None and self.test_dataset is None
            else self.test_dataset
        )
        return [(train_subset, dev_subset, test_subset)]

    def _split_leave_one(
        self, split_by: str
    ) -> List[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        """
        留一划分：以每个唯一的 split_by 作为测试集，
        在剩余中选一个作为验证集，其余作为训练集。
        """
        unique_ids = self.train_dataset.info[split_by].unique()
        splits = []
        for test_id in unique_ids:
            remaining_ids = [s for s in unique_ids if s != test_id]
            for val_id in remaining_ids:
                train_ids = [s for s in remaining_ids if s != val_id]
                train_subset = self._create_subset(train_ids, split_by)
                val_subset = self._create_subset([val_id], split_by)
                test_subset = self._create_subset([test_id], split_by)
                splits.append((train_subset, val_subset, test_subset))
        return splits

    def _stratified_split(
        self,
        ids: np.ndarray,
        labels: Optional[pd.Series],
        test_size: float,
        random_state: Optional[int],
        shuffle: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(sss.split(ids, labels))
        return ids[train_idx], ids[test_idx]

    def _random_split(
        self,
        ids: np.ndarray,
        labels: Optional[pd.Series],
        test_size: float,
        random_state: Optional[int],
        shuffle: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return train_test_split(
            ids, test_size=test_size, random_state=random_state, shuffle=shuffle
        )

    def _split_seedv(
        self, split_by: str, seed: Optional[int] = None
    ) -> List[Tuple[BaseDataset, BaseDataset, BaseDataset]]:
        """
        针对 SEED-V 数据集的划分方法，
        每个 session 的15个试验按顺序分为三个部分（5:5:5），
        并将所有 session 的结果合并得到训练集、验证集和测试集。
        """
        np.random.seed(seed)
        unique_sessions = self.train_dataset.info[split_by].unique()
        train_ids, val_ids, test_ids = [], [], []
        for session_id in unique_sessions:
            trials = [f"{session_id}_{i}" for i in range(15)]
            part1, part2, part3 = trials[:5], trials[5:10], trials[10:]
            train_ids.extend(part1)
            val_ids.extend(part2)
            test_ids.extend(part3)
        train_subset = self._create_subset(train_ids, "trial_id")
        val_subset = self._create_subset(val_ids, "trial_id")
        test_subset = self._create_subset(test_ids, "trial_id")
        return [(train_subset, val_subset, test_subset)]

    def _split_ninapro(
        self,
    ) -> List[Tuple[BaseDataset, BaseDataset, Optional[BaseDataset]]]:
        """
        针对 Ninapro 数据集的划分方法：
        根据 trial_id 划分，每个 trial 下要求有 6 个 stimulus_id，
        按 2:1 分割 stimulus_id 分别作为训练和测试。
        """
        unique_trial_ids = self.train_dataset.info["trial_id"].unique()
        train_indices, test_indices = [], []
        for trial_id in unique_trial_ids:
            trial_info = self.train_dataset.info[self.train_dataset.info["trial_id"] == trial_id]
            stimulus_ids = trial_info["stimulus_id"].unique()
            if len(stimulus_ids) != 6:
                raise ValueError(
                    f"Trial {trial_id} 的 stimulus_id 数量不是 6，而是 {len(stimulus_ids)}"
                )
            np.random.shuffle(stimulus_ids)
            split_point = int(len(stimulus_ids) * 2 / 3)
            train_stimuli = stimulus_ids[:split_point]
            test_stimuli = stimulus_ids[split_point:]
            train_indices.extend(train_stimuli)
            test_indices.extend(test_stimuli)
        train_subset = self._create_subset(train_indices, "stimulus_id")
        test_subset = self._create_subset(test_indices, "stimulus_id")
        return [(train_subset, test_subset, None)]
