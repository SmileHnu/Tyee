#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2026, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : k_fold.py
@Time    : 2026/03/01 15:30:47
@Desc    : KFold splitter
"""


import os
import re
import logging
import numpy as np
import pandas as pd
from copy import copy
from typing import Union, Tuple, Generator, List
from sklearn import model_selection

from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit


log = logging.getLogger('split')


class KFold(BaseSplit):
    def __init__(
        self,
        n_splits: int = 5,
        group_by: Union[None, str] = None,
        split_by: Union[None, str] = None,
        per_subject: bool = False,
        subject_key: str = 'subject_id',
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        dst_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize KFold splitter.

        Args:
            n_splits (int): Number of folds.
            group_by (Union[None, str]): Outer grouping column.
            split_by (Union[None, str]): Inner split unit column.
            per_subject (bool): Whether to split each subject independently.
            subject_key (str): Subject column name when per_subject is True.
            shuffle (bool): Whether to shuffle before split.
            random_state (Union[int, None]): Random seed.
            dst_path (Union[None, str]): Output directory for fold csv files.

        Mode selection:
            1) group_by=None, split_by=None:
               - per_subject=False -> global KFold on samples
               - per_subject=True  -> KFold on samples per subject
            2) group_by=None, split_by!=None:
               - per_subject=False -> KFold on unique split_by groups
               - per_subject=True  -> KFold on unique split_by groups per subject
            3) group_by!=None, split_by!=None:
               - per_subject=False -> for each group_by group, run KFold in each split_by subgroup and merge by fold
               - per_subject=True  -> same as above, but first split by subject_key
        """
        self.n_splits = n_splits
        self.group_by = group_by
        self.split_by = split_by
        self.per_subject = per_subject
        self.subject_key = subject_key
        self.shuffle = shuffle
        self.random_state = random_state
        self.dst_path = dst_path

        if self.dst_path is None:
            raise ValueError("`dst_path` must be provided for KFold split.")
        if n_splits < 2:
            raise ValueError(f"Number of splits must be at least 2, but got {n_splits}.")

        assert group_by in [None, 'subject_id', 'session_id', 'trial_id'], \
            f"KFold split does not support group_by {group_by}."
        assert split_by in [None, 'subject_id', 'session_id', 'trial_id'], \
            f"KFold split does not support split_by {split_by}."

        if group_by is not None and split_by is None:
            raise ValueError("When `group_by` is set, `split_by` must also be set.")

        self.k_fold = model_selection.KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def make_split_info(self, info: pd.DataFrame) -> None:
        """
        Create fold csv files according to current KFold mode.

        Args:
            info (pd.DataFrame): Dataset metadata table.
        """
        if self.per_subject:
            self._make_split_info_per_subject(info)
        else:
            self._make_split_info_global(info)

    def _make_split_info_global(self, info: pd.DataFrame) -> None:
        """
        Create fold files in global mode.

        Args:
            info (pd.DataFrame): Dataset metadata table.
        """
        if self.group_by is None and self.split_by is None:
            self._global_random_kfold(info)
        elif self.group_by is None and self.split_by is not None:
            self._global_cross_kfold(info, split_by=self.split_by)
        else:
            self._global_group_kfold(info, group_by=self.group_by, split_by=self.split_by)

    def _make_split_info_per_subject(self, info: pd.DataFrame) -> None:
        """
        Create fold files in per-subject mode.

        Args:
            info (pd.DataFrame): Dataset metadata table.
        """
        if self.subject_key not in info.columns:
            raise ValueError(f"`{self.subject_key}` column not found in dataset info.")

        subjects = sorted(set(info[self.subject_key]))
        for subject in subjects:
            subject_info = info[info[self.subject_key] == subject]

            if self.group_by is None and self.split_by is None:
                self._subject_random_kfold(subject_info, subject)
            elif self.group_by is None and self.split_by is not None:
                self._subject_cross_kfold(subject_info, subject, split_by=self.split_by)
            else:
                self._subject_group_kfold(subject_info, subject, group_by=self.group_by, split_by=self.split_by)

    def _global_random_kfold(self, info: pd.DataFrame) -> None:
        """
        Global random KFold split over rows.

        Args:
            info (pd.DataFrame): Dataset metadata table.
        """
        for fold_id, (train_index, val_index) in enumerate(self.k_fold.split(info)):
            train_info = info.iloc[train_index]
            val_info = info.iloc[val_index]
            self._save_global_fold(train_info, val_info, fold_id)

    def _global_cross_kfold(self, info: pd.DataFrame, split_by: str) -> None:
        """
        Global KFold split over unique split_by ids.

        Args:
            info (pd.DataFrame): Dataset metadata table.
            split_by (str): Split unit column.
        """
        group_ids = sorted(set(info[split_by]))
        for fold_id, (train_idx, val_idx) in enumerate(self.k_fold.split(group_ids)):
            train_group_ids = np.array(group_ids)[train_idx].tolist()
            val_group_ids = np.array(group_ids)[val_idx].tolist()

            train_info = pd.concat(
                [info[info[split_by] == gid] for gid in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [info[info[split_by] == gid] for gid in val_group_ids],
                ignore_index=True
            )
            self._save_global_fold(train_info, val_info, fold_id)

    def _global_group_kfold(self, info: pd.DataFrame, group_by: str, split_by: str) -> None:
        """
        Global grouped KFold split.

        Args:
            info (pd.DataFrame): Dataset metadata table.
            group_by (str): Outer group column.
            split_by (str): Inner split unit column.
        """
        base_groups = sorted(set(info[group_by]))
        fold_train_infos = {i: [] for i in range(self.n_splits)}
        fold_val_infos = {i: [] for i in range(self.n_splits)}

        for base_group in base_groups:
            base_group_info = info[info[group_by] == base_group]
            split_ids = sorted(set(base_group_info[split_by]))

            for split_id in split_ids:
                split_info = base_group_info[base_group_info[split_by] == split_id]
                for fold_id, (train_idx, val_idx) in enumerate(self.k_fold.split(split_info)):
                    fold_train_infos[fold_id].append(split_info.iloc[train_idx])
                    fold_val_infos[fold_id].append(split_info.iloc[val_idx])

        for fold_id in range(self.n_splits):
            train_info = pd.concat(fold_train_infos[fold_id], ignore_index=True)
            val_info = pd.concat(fold_val_infos[fold_id], ignore_index=True)
            self._save_global_fold(train_info, val_info, fold_id)

    def _subject_random_kfold(self, subject_info: pd.DataFrame, subject: Union[str, int]) -> None:
        """
        Random KFold split for one subject.

        Args:
            subject_info (pd.DataFrame): Subject subset table.
            subject (Union[str, int]): Subject identifier.
        """
        for fold_id, (train_idx, val_idx) in enumerate(self.k_fold.split(subject_info)):
            train_info = subject_info.iloc[train_idx]
            val_info = subject_info.iloc[val_idx]
            self._save_subject_fold(train_info, val_info, subject, fold_id)

    def _subject_cross_kfold(self, subject_info: pd.DataFrame, subject: Union[str, int], split_by: str) -> None:
        """
        KFold split for one subject by split_by ids.

        Args:
            subject_info (pd.DataFrame): Subject subset table.
            subject (Union[str, int]): Subject identifier.
            split_by (str): Split unit column.
        """
        split_ids = sorted(set(subject_info[split_by]))
        for fold_id, (train_idx, val_idx) in enumerate(self.k_fold.split(split_ids)):
            train_group_ids = np.array(split_ids)[train_idx].tolist()
            val_group_ids = np.array(split_ids)[val_idx].tolist()

            train_info = pd.concat(
                [subject_info[subject_info[split_by] == gid] for gid in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [subject_info[subject_info[split_by] == gid] for gid in val_group_ids],
                ignore_index=True
            )
            self._save_subject_fold(train_info, val_info, subject, fold_id)

    def _subject_group_kfold(
        self,
        subject_info: pd.DataFrame,
        subject: Union[str, int],
        group_by: str,
        split_by: str,
    ) -> None:
        """
        Grouped KFold split for one subject.

        Args:
            subject_info (pd.DataFrame): Subject subset table.
            subject (Union[str, int]): Subject identifier.
            group_by (str): Outer group column.
            split_by (str): Inner split unit column.
        """
        base_groups = sorted(set(subject_info[group_by]))
        fold_train_infos = {i: [] for i in range(self.n_splits)}
        fold_val_infos = {i: [] for i in range(self.n_splits)}

        for base_group in base_groups:
            base_group_info = subject_info[subject_info[group_by] == base_group]
            split_ids = sorted(set(base_group_info[split_by]))

            for split_id in split_ids:
                split_info = base_group_info[base_group_info[split_by] == split_id]
                for fold_id, (train_idx, val_idx) in enumerate(self.k_fold.split(split_info)):
                    fold_train_infos[fold_id].append(split_info.iloc[train_idx])
                    fold_val_infos[fold_id].append(split_info.iloc[val_idx])

        for fold_id in range(self.n_splits):
            train_info = pd.concat(fold_train_infos[fold_id], ignore_index=True)
            val_info = pd.concat(fold_val_infos[fold_id], ignore_index=True)
            self._save_subject_fold(train_info, val_info, subject, fold_id)

    def _save_global_fold(self, train_info: pd.DataFrame, val_info: pd.DataFrame, fold_id: int) -> None:
        """
        Save one global fold split to csv files.

        Args:
            train_info (pd.DataFrame): Train table.
            val_info (pd.DataFrame): Validation table.
            fold_id (int): Fold id.
        """
        train_info.to_csv(os.path.join(self.dst_path, f'train_fold_{fold_id}.csv'), index=False)
        val_info.to_csv(os.path.join(self.dst_path, f'val_fold_{fold_id}.csv'), index=False)

    def _save_subject_fold(
        self,
        train_info: pd.DataFrame,
        val_info: pd.DataFrame,
        subject: Union[str, int],
        fold_id: int,
    ) -> None:
        """
        Save one subject fold split to csv files.

        Args:
            train_info (pd.DataFrame): Train table.
            val_info (pd.DataFrame): Validation table.
            subject (Union[str, int]): Subject identifier.
            fold_id (int): Fold id.
        """
        train_info.to_csv(os.path.join(self.dst_path, f'train_subject_{subject}_fold_{fold_id}.csv'), index=False)
        val_info.to_csv(os.path.join(self.dst_path, f'val_subject_{subject}_fold_{fold_id}.csv'), index=False)

    @property
    def subjects(self) -> List[str]:
        """
        Get available subject ids from per-subject fold files.

        Returns:
            List[str]: Sorted subject id list.
        """
        indice_files = os.listdir(self.dst_path)

        def indice_file_to_subject(indice_file: str) -> Union[str, None]:
            match = re.search(r'subject_(.*?)_fold_\d+\.csv', indice_file)
            return match.group(1) if match else None

        subjects = list(set(map(indice_file_to_subject, indice_files)))
        subjects = [s for s in subjects if s is not None]
        subjects = sorted(subjects)
        return subjects

    @property
    def fold_ids(self) -> List[int]:
        """
        Get available fold ids from generated fold files.

        Returns:
            List[int]: Sorted fold id list.
        """
        indice_files = os.listdir(self.dst_path)

        def indice_file_to_fold_id(indice_file: str) -> Union[int, None]:
            match = re.search(r'fold_(\d+)\.csv', indice_file)
            return int(match.group(1)) if match else None

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids = [f for f in fold_ids if f is not None]
        fold_ids = sorted(fold_ids)
        return fold_ids

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        subject: Union[int, None] = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Yield fold datasets from saved csv files.

        Args:
            dataset (BaseDataset): Source dataset object.
            val_dataset (BaseDataset): Reserved argument.
            test_dataset (BaseDataset): Reserved argument.
            subject (Union[int, None]): Subject filter when per_subject is True.

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]:
            train, val, test datasets.
        """
        if not self.check_dst_path():
            log.info('📊 | Creating the split of train and val datasets.')
            log.info(
                f'😊 | Please set \033[92mdst_path\033[0m to \033[92m{self.dst_path}\033[0m '
                'for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.dst_path, exist_ok=True)
            self.make_split_info(dataset.info)
        else:
            log.info(
                f'📊 | Detected existing split of train and val sets. Using existing split from {self.dst_path}.'
            )
            log.info(
                '💡 | If the dataset is re-generated, you need to re-generate the split instead of using the previous split.'
            )

        if self.per_subject:
            subjects = self.subjects
            fold_ids = self.fold_ids

            if subject is not None:
                subject = str(subject)
                assert subject in subjects, f'The subject should be in the subject list {subjects}.'

            for local_subject in subjects:
                if subject is not None and local_subject != subject:
                    continue
                for fold_id in fold_ids:
                    train_info = pd.read_csv(
                        os.path.join(self.dst_path, f'train_subject_{local_subject}_fold_{fold_id}.csv'),
                        low_memory=False
                    )
                    val_info = pd.read_csv(
                        os.path.join(self.dst_path, f'val_subject_{local_subject}_fold_{fold_id}.csv'),
                        low_memory=False
                    )

                    train_dataset = copy(dataset)
                    train_dataset.info = train_info

                    val_dataset = copy(dataset)
                    val_dataset.info = val_info

                    yield train_dataset, val_dataset, test_dataset
        else:
            for fold_id in self.fold_ids:
                train_info = pd.read_csv(
                    os.path.join(self.dst_path, f'train_fold_{fold_id}.csv'),
                    low_memory=False
                )
                val_info = pd.read_csv(
                    os.path.join(self.dst_path, f'val_fold_{fold_id}.csv'),
                    low_memory=False
                )

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                val_dataset = copy(dataset)
                val_dataset.info = val_info

                yield train_dataset, val_dataset, test_dataset

    @property
    def repr_body(self) -> dict:
        """
        Build dict used by repr.

        Returns:
            dict: Representation fields.
        """
        return {
            'n_splits': self.n_splits,
            'group_by': self.group_by,
            'split_by': self.split_by,
            'per_subject': self.per_subject,
            'subject_key': self.subject_key,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'dst_path': self.dst_path
        }

    def __repr__(self) -> str:
        """
        Return formatted string representation.

        Returns:
            str: String representation.
        """
        format_string = f"{self.__class__.__name__}("
        format_string += ', '.join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in self.repr_body.items()
        )
        format_string += ')'
        return format_string
