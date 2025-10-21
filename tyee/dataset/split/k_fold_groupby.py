import logging
import os
import re
from copy import copy
from typing import Dict, List, Tuple, Union, Generator

import pandas as pd
from sklearn import model_selection

from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit

log = logging.getLogger('split')


class KFoldGroupby(BaseSplit):
    def __init__(
        self,
        group_by: str,
        base_group: str = 'subject_id',
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the KFoldGroupbyTrial class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id').
            base_group (str): The column name for the base grouping (e.g., 'subject_id', 'session_id').
            n_splits (int): Number of folds for splitting.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (Union[int, None]): Random seed for reproducibility.
            split_path (Union[None, str]): Path to save split files.
        """
        self.group_by = group_by
        self.base_group = base_group
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        # Initialize KFold from sklearn
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
        base_groups = sorted(set(info[self.base_group]))

        train_infos = {}
        val_infos = {}

        for base_group in base_groups:
            base_group_info = info[info[self.base_group] == base_group]
            group_ids = sorted(set(base_group_info[self.group_by]))

            for group_id in group_ids:
                group_info = base_group_info[base_group_info[self.group_by] == group_id]

                for i, (train_index, val_index) in enumerate(self.k_fold.split(group_info)):
                    train_info = group_info.iloc[train_index]
                    val_info = group_info.iloc[val_index]

                    if i not in train_infos:
                        train_infos[i] = []
                    if i not in val_infos:
                        val_infos[i] = []

                    train_infos[i].append(train_info)
                    val_infos[i].append(val_info)

        for i in train_infos.keys():
            train_info = pd.concat(train_infos[i], ignore_index=True)
            val_info = pd.concat(val_infos[i], ignore_index=True)

            train_info.to_csv(
                os.path.join(self.split_path, f'train_fold_{i}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_fold_{i}.csv'),
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
        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
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
                os.path.join(self.split_path, f'train_fold_{fold_id}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_fold_{fold_id}.csv')
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