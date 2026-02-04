#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2026, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : hold_out.py
@Time    : 2026/02/03 23:52:25
@Desc    : 
"""
import os
import logging
import numpy as np
import pandas as pd
from copy import copy
from typing import Union, Tuple, Generator, List
from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit


log = logging.getLogger('split')


class HoldOut(BaseSplit):
    def __init__(
        self,
        rsize: List[Union[float, int]] = [0.8, 0.2, 0.0],
        rtype: str = 'ratio',
        group_by: Union[None, str] = None,
        split_by: Union[None, str] = None,
        shuffle: bool = False,
        stratify: str = None,
        random_state: Union[int, None] = None,
        dst_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOut class with the specified parameters.
  
        :param List[Union[float, int]] rsize: array of number or ratio of dataset split, defaults to [0.8, 0.2, 0.0]
        :param str rtype: type of dataset split, either 'ratio' or 'number', defaults to 'ratio'
        :param Union[None, str] group_by: column name to group by, defaults to None
        :param Union[None, str] split_by: column name to split by, defaults to None
        :param bool shuffle: whether to shuffle the data before splitting, defaults to False
        :param str stratify: column name to stratify by which balanced the ratio of column, defaults to None
        :param Union[int, None] random_state: Random seed for reproducibility, defaults to None
        :param Union[None, str] dst_path: path to save split files. If None, a random path will be generated., defaults to None
        """
        self.rsize = rsize
        self.rtype = rtype
        
        self.group_by = group_by
        self.split_by = split_by
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_state = random_state
        self.dst_path = dst_path

        assert len(rsize) == 3, "rsize should be a list of three elements" 
        if rtype == 'ratio':
            assert sum(rsize) == 1.0, "When rtype is 'ratio', the sum of rsize should be 1.0"
        elif rtype == 'number':
            assert all(isinstance(x, int) and x >= 0 for x in rsize), \
                "When rtype is 'number', all elements in rsize should be non-negative integers"
        else:
            raise ValueError(f"rtype should be either 'ratio' or 'number', but got {rtype}.")

        assert group_by in [None, "subject_id", "session_id", "trial_id"], \
            f"HoldOut split does not support group_by {group_by}."
        assert split_by in [None, "subject_id", "session_id", "trial_id"], \
            f"HoldOut split does not support group_by {split_by}."

    def make_split_info(self, info: pd.DataFrame) -> None:
        """
        Create train, val, test split files.

        :param pd.DataFrame info: DataFrame containing dataset information.
        """

        # random split
        if self.group_by is None and self.split_by is None:
            train_info, val_info, test_info = self._random_split(
                info,
                self.rsize,
                self.rtype,
                self.random_state,
                self.shuffle,
                self.stratify
            )
        elif self.split_by is not None and self.group_by is None:
            self._by_split(
                info=info,
                rsize=self.rsize,
                rtype=self.rtype,
                split_by=self.split_by,
                group_by=self.group_by,
                random_seed=self.random_state,
                shuffle=self.shuffle,
            )
        elif self.group_by is not None:
            self._group_split(
                info=info,
                rsize=self.rsize,
                rtype=self.rtype,
                split_by=self.split_by,
                group_by=self.group_by,
                random_seed=self.random_state,
                shuffle=self.shuffle,
            )
        else:
            raise NotImplementedError(
                f"HoldOut split does not support group_by {self.group_by} and split_by {self.split_by}."
            )

        train_info.to_csv(os.path.join(self.dst_path, 'train.csv'), index=False)
        if val_info is not None:
            val_info.to_csv(os.path.join(self.dst_path, 'val.csv'), index=False)
        if test_info is not None:
            test_info.to_csv(os.path.join(self.dst_path, 'test.csv'), index=False)

    def _random_split(
            self,
            info: pd.DataFrame,
            rsize: List[float| int],
            random_seed: int, 
            shuffle: bool, 
            stratify: Union[None, str] = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Randomly split the dataset into train and val sets.

        :param pd.DataFrame info: DataFrame containing dataset information.
        :param List[float| int] rsize: array of number or ratio of dataset split, defaults to [0.8, 0.2, 0.0]
        :param bool shuffle: whether to shuffle the data before splitting, defaults to False
        :param str stratify: column name to stratify by which balanced the ratio of column, defaults to None
        :param int random_state: Random seed for reproducibility, defaults to None
        """
        test_info = None

        if shuffle:
            info = info.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Do split  
        if stratify is not None and stratify in info.columns:
            train_indices = []
            val_indices = []
            test_indices = []
            r_train, r_val, r_test = self._to_ratio(rsize, len(info))

            for group_label, group_df in info.groupby(stratify, sort=False): 
                n_samples_in_group = len(group_df)

                n_val_group = int(round(n_samples_in_group * r_val))
                n_test_group = int(round(n_samples_in_group * r_test))
                n_train_group = n_samples_in_group - n_val_group - n_test_group
                
                assert n_train_group > 0, "num_train is too small to split with the given stratify column."
                assert n_val_group > 0, "num_val is too small to split with the given stratify column."
                assert n_test_group > 0, "num_test is too small to split with the given stratify column."
            
                group_original_indices = group_df.index.tolist()
                train_indices.extend(group_original_indices[:n_train_group])
                val_indices.extend(group_original_indices[n_train_group:n_train_group + n_val_group])
                test_indices.extend(group_original_indices[n_train_group + n_val_group:])

            train_info = info.loc[train_indices].copy()
            val_info = info.loc[val_indices].copy()
            if r_test > 0:
                test_info = info.loc[test_indices].copy()
        else:

            n_train, n_val, n_test = self._to_number(rsize, len(info))
            train_info = info.iloc[:n_train].copy()
            val_info = info.iloc[n_train:n_train + n_val].copy()
            if n_test > 0:
                test_info = info.iloc[n_train + n_val:].copy()                
        return train_info, val_info, test_info
    
    def _group_split(self,
            info: pd.DataFrame,
            rsize: List[float| int],
            split_by: str,
            group_by: Union[None, str],
            random_seed: int, 
            shuffle: bool, 
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Grouped split the dataset into train, val and test sets.

        :param pd.DataFrame info: DataFrame containing dataset information.
        :param List[float| int] rsize: array of number or ratio of dataset split, defaults to [0.8, 0.2, 0.0]
        :param Union[None, str] split_by: column name to split by, defaults to None
        :param Union[None, str] group_by: column name to group by, defaults to None
        :param bool shuffle: whether to shuffle the data before splitting, defaults to False
        :param Union[None, str] stratify: column name to stratify by which balanced the ratio of column, defaults to None
        :param int random_state: Random seed for reproducibility, defaults to None
        """
        combined_test_info = None
        
        assert group_by is not None, "group_by should not be None for group split."
        assert split_by is not None, "split_by should not be None for group split."
        # Extract all unique base groups
        base_groups = sorted(set(info[group_by]))

        # Initialize lists to store train and validation DataFrames
        train_infos = []
        val_infos = []
        test_infos = []

        # Process each base group for loop
        for base_group in base_groups:
            # Filter rows belonging to the current base group
            base_group_info = info[info[group_by] == base_group]

            # Extract unique secondary groups within the base group
            group_ids = sorted(set(base_group_info[split_by]))

            r_train, r_val, t_test = self._to_ratio(rsize)
            if shuffle:
                np.random.seed(random_seed)
                np.random.shuffle(group_ids)
            
            n_val_groups = int(len(group_ids) * r_val)
            n_test_groups = int(len(group_ids) * t_test)
            n_train_groups = len(group_ids) - n_val_groups - n_test_groups
            train_group_ids = group_ids[:n_train_groups]
            val_group_ids = group_ids[n_train_groups:n_train_groups + n_val_groups]
            test_group_ids = group_ids[n_train_groups + n_val_groups:]

            # Assign rows to training and validation sets based on the split group IDs
            train_info = pd.concat(
                [base_group_info[base_group_info[split_by] == by_id] for by_id in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [base_group_info[base_group_info[split_by] == by_id] for by_id in val_group_ids],
                ignore_index=True
            )
            test_info = pd.concat(
                [base_group_info[base_group_info[split_by] == by_id] for by_id in test_group_ids],
                ignore_index=True
            )

            # Append the results to the lists
            train_infos.append(train_info)
            val_infos.append(val_info)
            test_infos.append(test_info)

        # Step 3: Combine all base groups' train and validation sets
        combined_train_info = pd.concat(train_infos, ignore_index=True)
        combined_val_info = pd.concat(val_infos, ignore_index=True)
        if rsize[-1] > 0:
            combined_test_info = pd.concat(test_infos, ignore_index=True)

        return combined_train_info, combined_val_info, combined_test_info

    def _by_split(self,
            info: pd.DataFrame,
            rsize: List[float| int],
            split_by: str,
            random_seed: int, 
            shuffle: bool, 
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        split the dataset into train, val and test sets by `split_by`.

        :param pd.DataFrame info: DataFrame containing dataset information.
        :param List[float| int] rsize: array of number or ratio of dataset split, defaults to [0.8, 0.2, 0.0]
        :param Union[None, str] split_by: column name to split by, defaults to None
        :param bool shuffle: whether to shuffle the data before splitting, defaults to False
        :param Union[None, str] stratify: column name to stratify by which balanced the ratio of column, defaults to None
        :param int random_state: Random seed for reproducibility, defaults to None
        """
        test_info = None
        
        r_train, r_val, r_test = self._to_ratio(rsize, len(info))
        all_group_ids = list(info[split_by].unique())

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(all_group_ids)

        n_val_groups = int(len(all_group_ids) * r_val)
        n_test_groups = int(len(all_group_ids) * r_test)
        n_train_groups = len(all_group_ids) - n_val_groups - n_test_groups
        train_group_ids = all_group_ids[:n_train_groups]
        val_group_ids = all_group_ids[n_train_groups:n_train_groups + n_val_groups]
        test_group_ids = all_group_ids[n_train_groups + n_val_groups:]

        # Create train, val and test dataframes
        train_info = pd.concat(
            [info[info[split_by] == by_id] for by_id in train_group_ids],
            ignore_index=True
        )
        val_info = pd.concat(
            [info[info[split_by] == by_id] for by_id in val_group_ids],
            ignore_index=True
        )
        if r_test > 0:
            test_info = pd.concat(
                [info[info[split_by] == by_id] for by_id in test_group_ids],
                ignore_index=True
            )
        
        return train_info, val_info, test_info

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train, val and test datasets.

        :param BaseDataset dataset: The dataset to split.
        :return Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]: Generator yielding train, val, and test datasets.
        """
        if not self.check_dst_path():
            log.info('📊 | Creating the split of train, val and test datasets.')
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

        # Load train and val splits
        train_info = pd.read_csv(os.path.join(self.dst_path, 'train.csv'))
        val_info = pd.read_csv(os.path.join(self.dst_path, 'val.csv'))

        # Create train and val datasets
        train_dataset = copy(dataset)
        train_dataset.info = train_info

        val_dataset = copy(dataset)
        val_dataset.info = val_info

        if self.rsize[-1] > 0:
            test_info = pd.read_csv(os.path.join(self.dst_path, 'test.csv'))
            test_dataset = copy(dataset)
            test_dataset.info = test_info

        yield train_dataset, val_dataset, test_dataset

    @property
    def repr_body(self) -> dict:
        """
        Representation body for the class.

        :return dict: Dictionary containing class attributes.
        """
        return {
            'rsize': self.rsize,
            'rtype': self.rtype,
            'group_by': self.group_by,
            'split_by': self.split_by,
            'shuffle': self.shuffle,
            'stratify': self.stratify,
            'random_state': self.random_state,
            'dst_path': self.dst_path
        }

    def __repr__(self) -> str:
        """
        String representation of the class.

        :return str: Formatted string representation.
        """
        format_string = f"{self.__class__.__name__}("
        format_string += ', '.join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in self.repr_body.items()
        )
        format_string += ')'
        return format_string