import logging
import os
import re
from copy import copy
from typing import Dict, Tuple, Union, Generator, List

import numpy as np
import pandas as pd
from sklearn import model_selection

from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class KFoldCross(BaseSplit):
    def __init__(
        self,
        group_by: str,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the KFoldCross class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id', 'subject_id').
            n_splits (int): Number of folds for splitting.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (Union[None, int]): Random seed for reproducibility.
            split_path (Union[None, str]): Path to save split files.
        """
        self.group_by = group_by
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        if n_splits < 2:
            raise ValueError(
                f'Number of splits must be at least 2, but got {n_splits}.'
            )

        # Initialize KFold
        self.k_fold = model_selection.KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each fold.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        group_ids = sorted(set(info[self.group_by]))

        for fold_id, (train_index_group_ids, val_index_group_ids) in enumerate(self.k_fold.split(group_ids)):
            if len(train_index_group_ids) == 0 or len(val_index_group_ids) == 0:
                raise ValueError(
                    f'The number of training or validation groups is zero for fold {fold_id}.'
                )

            train_group_ids = np.array(group_ids)[train_index_group_ids].tolist()
            val_group_ids = np.array(group_ids)[val_index_group_ids].tolist()

            train_info = pd.concat(
                [info[info[self.group_by] == train_group_id] for train_group_id in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [info[info[self.group_by] == val_group_id] for val_group_id in val_group_ids],
                ignore_index=True
            )

            train_info.to_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_fold_{fold_id}.csv'),
                index=False
            )

    @property
    def fold_ids(self) -> List[int]:
        """
        Retrieve the list of fold IDs based on existing split files.

        Returns:
            List[int]: Sorted list of fold IDs.
        """
        indice_files = os.listdir(self.split_path)

        def indice_file_to_fold_id(indice_file: str) -> Union[int, None]:
            # Extract fold ID from file name using regex
            match = re.search(r'fold_(\d+).csv', indice_file)
            return int(match.group(1)) if match else None

        # Filter and sort unique fold IDs
        fold_ids = sorted(set(filter(None, map(indice_file_to_fold_id, indice_files))))
        return fold_ids

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each fold.

        Args:
            dataset (BaseDataset): The dataset to split.

        Yields:
            Tuple[BaseDataset, BaseDataset]: Train and validation datasets for each fold.
        """
        if not self.check_split_path():
            log.info('📊 | Creating the split of train and validation sets.')
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m '
                'for the next run, if you want to use the same setting for the experiment.'
            )
            os.makedirs(self.split_path)
            self.split_info_constructor(dataset.info)
        else:
            log.info(f'📊 | Detected existing split from {self.split_path}.')
            log.info(
                '💡 | If the dataset is re-generated, re-generate the split instead of using the previous one.'
            )

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv'), low_memory=False
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_fold_{fold_id}.csv'), low_memory=False
            )

            train_dataset = copy(dataset)
            train_dataset.info = train_info

            val_dataset = copy(dataset)
            val_dataset.info = val_info

            yield train_dataset, val_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        """
        Representation body for the class.

        Returns:
            Dict: Dictionary containing class attributes.
        """
        return {
            'group_by': self.group_by,
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns:
            str: Formatted string representation.
        """
        format_string = f"{self.__class__.__name__}("
        format_string += ', '.join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in self.repr_body.items()
        )
        format_string += ')'
        return format_string