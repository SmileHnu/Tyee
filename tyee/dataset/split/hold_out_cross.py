import logging
import os
import re
from copy import copy
from typing import Dict, Tuple, Union, Generator, List

import numpy as np
import pandas as pd
from sklearn import model_selection

from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit

log = logging.getLogger('split')


class HoldOutCross(BaseSplit):
    """
    A class for performing hold-out cross-validation based on a specific grouping column.

    This class splits the dataset into training and validation sets by grouping the data
    based on a specified column (e.g., 'trial_id', 'session_id', 'subject_id'). The split
    ensures that all rows belonging to the same group are either in the training set or
    the validation set, but not both.

    Attributes:
        group_by (str): The column name used to group the data for splitting.
        val_size (float): The proportion of the dataset to include in the validation split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (Union[None, int]): Random seed for reproducibility.
        split_path (Union[None, str]): Path to save the split files.
    """

    def __init__(
        self,
        group_by: str,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOutCross class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id', 'subject_id').
            val_size (float): Proportion of the dataset to include in the validation split. Should be between 0.0 and 1.0. (default: 0.2)
            shuffle (bool): Whether to shuffle the data before splitting. (default: False)
            random_state (Union[None, int]): Random seed for reproducibility. (default: None)
            split_path (Union[None, str]): Path to save split files. If None, splits are not saved. (default: None)
        """
        self.group_by = group_by
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files based on the specified grouping column.

        This method groups the dataset by the specified column (`group_by`) and performs
        a train-validation split on the unique group IDs. It ensures that all rows
        belonging to the same group are either in the training set or the validation set.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information. This DataFrame
                                 must include the column specified by `group_by`.

        Steps:
            1. Extract unique group IDs from the `group_by` column.
            2. Perform a train-validation split on the group IDs using `train_test_split`.
            3. Create training and validation DataFrames by filtering rows based on the split group IDs.
            4. Save the resulting DataFrames as CSV files (`train.csv` and `val.csv`) in the specified `split_path`.
        """
        # Step 1: Extract unique group IDs
        group_ids = sorted(set(info[self.group_by]))
        

        # Step 2: Perform train-validation split on group IDs
        train_group_ids, val_group_ids = model_selection.train_test_split(
            group_ids,
            test_size=self.val_size,
            random_state=self.random_state,
            shuffle=self.shuffle
        )

        # Step 3: Create training and validation DataFrames
        train_info = pd.concat(
            [info[info[self.group_by] == group_id] for group_id in train_group_ids],
            ignore_index=True
        )
        val_info = pd.concat(
            [info[info[self.group_by] == group_id] for group_id in val_group_ids],
            ignore_index=True
        )

        # Step 4: Save the resulting DataFrames as CSV files
        train_info.to_csv(
            os.path.join(self.split_path, 'train.csv'),
            index=False
        )
        val_info.to_csv(
            os.path.join(self.split_path, 'val.csv'),
            index=False
        )

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets based on the constructed split files.

        This method checks if the split files (`train.csv` and `val.csv`) exist in the
        specified `split_path`. If they do not exist, it calls `split_info_constructor`
        to create them. It then loads the split files and creates corresponding
        `BaseDataset` objects for the training and validation sets.

        Args:
            dataset (BaseDataset): The dataset to split.
            val_dataset (BaseDataset): Not used in this implementation.
            test_dataset (BaseDataset): The test dataset (optional).

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing the training dataset,
                                                          validation dataset, and test dataset.

        Steps:
            1. Check if the split files exist in `split_path`.
               - If not, call `split_info_constructor` to create them.
            2. Load the split files (`train.csv` and `val.csv`) into DataFrames.
            3. Create shallow copies of the input dataset and assign the loaded DataFrames
               to their `info` attributes.
            4. Yield the training dataset, validation dataset, and test dataset.
        """
        # Step 1: Check if split files exist
        if not self.check_split_path():
            log.info('📊 | Creating the split of train and val sets.')
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

        # Step 2: Load the split files
        train_info = pd.read_csv(
            os.path.join(self.split_path, 'train.csv'), low_memory=False
        )
        val_info = pd.read_csv(
            os.path.join(self.split_path, 'val.csv'), low_memory=False
        )

        # Step 3: Create shallow copies of the dataset
        train_dataset = copy(dataset)
        train_dataset.info = train_info

        val_dataset = copy(dataset)
        val_dataset.info = val_info

        # Step 4: Yield the datasets
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
            'val_size': self.val_size,
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


class HoldOutCrossByIds(BaseSplit):
    """
    A class for performing hold-out validation by specifying the exact group IDs for the validation set.

    This class splits the dataset into training and validation sets by grouping the data
    based on a specified column (e.g., 'trial_id', 'session_id', 'subject_id'). The split
    is determined by a provided list of group IDs, which will constitute the validation set.
    All other groups will form the training set.

    Attributes:
        group_by (str): The column name used to group the data for splitting.
        val_ids (List[Union[str, int]]): A list of group IDs to be used for the validation set.
        split_path (Union[None, str]): Path to save the split files.
    """

    def __init__(
        self,
        group_by: str,
        val_ids: List[Union[str, int]],
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOutCrossByIds class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id', 'subject_id').
            val_ids (List[Union[str, int]]): A list of group IDs to be included in the validation set.
            split_path (Union[None, str]): Path to save split files. If None, splits are not saved. (default: None)
        """
        self.group_by = group_by
        self.val_ids = val_ids
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files based on the specified validation IDs.

        This method splits the dataset into training and validation sets based on the provided `val_ids` list.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information. This DataFrame
                                 must include the column specified by `group_by`.
        """
        # Step 1: Get all unique group IDs
        all_group_ids = set(info[self.group_by])
        
        # Step 2: The validation group IDs are the ones provided
        val_group_ids = self.val_ids
        
        # The training group IDs are the ones not in the validation set
        train_group_ids = list(all_group_ids - set(val_group_ids))

        # Step 3: Create training and validation DataFrames
        train_info = pd.concat(
            [info[info[self.group_by] == group_id] for group_id in train_group_ids],
            ignore_index=True
        )
        val_info = pd.concat(
            [info[info[self.group_by] == group_id] for group_id in val_group_ids],
            ignore_index=True
        )

        # Step 4: Save the resulting DataFrames as CSV files
        train_info.to_csv(
            os.path.join(self.split_path, 'train.csv'),
            index=False
        )
        val_info.to_csv(
            os.path.join(self.split_path, 'val.csv'),
            index=False
        )

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets based on the constructed split files.
        """
        # Step 1: Check if split files exist
        if not self.check_split_path():
            log.info('📊 | Creating the split of train and val sets.')
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

        # Step 2: Load the split files
        train_info = pd.read_csv(
            os.path.join(self.split_path, 'train.csv'), low_memory=False
        )
        val_info = pd.read_csv(
            os.path.join(self.split_path, 'val.csv'), low_memory=False
        )

        # Step 3: Create shallow copies of the dataset
        train_dataset = copy(dataset)
        train_dataset.info = train_info

        val_dataset = copy(dataset)
        val_dataset.info = val_info

        # Step 4: Yield the datasets
        yield train_dataset, val_dataset, test_dataset

    @property
    def repr_body(self) -> Dict:
        """
        Representation body for the class.
        """
        return {
            'group_by': self.group_by,
            'val_ids': self.val_ids,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        """
        String representation of the class.
        """
        format_string = f"{self.__class__.__name__}("
        format_string += ', '.join(
            f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
            for k, v in self.repr_body.items()
        )
        format_string += ')'
        return format_string