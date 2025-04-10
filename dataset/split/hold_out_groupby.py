import logging
import os
from copy import copy
from typing import Dict, List, Tuple, Union, Generator

import pandas as pd
from sklearn import model_selection

from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class HoldOutGroupby(BaseSplit):
    """
    A class for performing hold-out validation with hierarchical grouping.

    This class splits the dataset into training and validation sets by grouping the data
    based on two levels of grouping:
    1. A base grouping column (e.g., 'subject_id', 'session_id').
    2. A secondary grouping column (e.g., 'trial_id', 'session_id').

    The split ensures that all rows belonging to the same group (defined by `group_by`)
    within a base group (defined by `base_group`) are either in the training set or the
    validation set, but not both. This hierarchical grouping is useful for datasets
    where data is nested (e.g., trials within subjects).

    Attributes:
        group_by (str): The column name used for secondary grouping (e.g., 'trial_id').
        base_group (str): The column name used for base grouping (e.g., 'subject_id').
        val_size (float): The proportion of the dataset to include in the validation split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (Union[int, None]): Random seed for reproducibility.
        split_path (Union[None, str]): Path to save the split files.
    """

    def __init__(
        self,
        group_by: str,
        base_group: str = "subject_id",
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOutGroupby class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id').
            base_group (str): The column name for the base grouping (e.g., 'subject_id', 'session_id').
            val_size (float): Proportion of the dataset to include in the validation split. Should be between 0.0 and 1.0. (default: 0.2)
            shuffle (bool): Whether to shuffle the data before splitting. (default: False)
            random_state (Union[int, None]): Random seed for reproducibility. (default: None)
            split_path (Union[None, str]): Path to save split files. (default: None)
        """
        self.group_by = group_by
        self.base_group = base_group
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each base group.

        This method performs the following steps:
        1. Extract all unique base groups from the `base_group` column.
        2. For each base group:
           - Extract all unique secondary groups (`group_by`) within the base group.
           - Perform a train-validation split on the secondary groups.
           - Assign rows belonging to the training and validation groups to separate DataFrames.
        3. Combine the training and validation DataFrames from all base groups.
        4. Save the combined training and validation DataFrames as CSV files.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information. This DataFrame
                                 must include the columns specified by `group_by` and `base_group`.
        """
        # Step 1: Extract all unique base groups
        base_groups = list(set(info[self.base_group]))

        # Initialize lists to store train and validation DataFrames
        train_infos = []
        val_infos = []

        # Step 2: Process each base group
        for base_group in base_groups:
            # Filter rows belonging to the current base group
            base_group_info = info[info[self.base_group] == base_group]

            # Extract unique secondary groups within the base group
            group_ids = list(set(base_group_info[self.group_by]))

            # Perform train-validation split on the secondary groups
            train_group_ids, val_group_ids = model_selection.train_test_split(
                group_ids,
                test_size=self.val_size,
                random_state=self.random_state,
                shuffle=self.shuffle
            )

            # Assign rows to training and validation sets based on the split group IDs
            train_info = pd.concat(
                [base_group_info[base_group_info[self.group_by] == group_id] for group_id in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [base_group_info[base_group_info[self.group_by] == group_id] for group_id in val_group_ids],
                ignore_index=True
            )

            # Append the results to the lists
            train_infos.append(train_info)
            val_infos.append(val_info)

        # Step 3: Combine all base groups' train and validation sets
        combined_train_info = pd.concat(train_infos, ignore_index=True)
        combined_val_info = pd.concat(val_infos, ignore_index=True)

        # Step 4: Save the combined train and validation sets to CSV files
        combined_train_info.to_csv(
            os.path.join(self.split_path, 'train.csv'),
            index=False
        )
        combined_val_info.to_csv(
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

        This method performs the following steps:
        1. Check if the split files (`train.csv` and `val.csv`) exist in the specified `split_path`.
           - If not, call `split_info_constructor` to create them.
        2. Load the split files into DataFrames.
        3. Create shallow copies of the input dataset and assign the loaded DataFrames
           to their `info` attributes.
        4. Yield the training dataset, validation dataset, and test dataset.

        Args:
            dataset (BaseDataset): The dataset to split.
            val_dataset (BaseDataset): Not used in this implementation.
            test_dataset (BaseDataset): The test dataset (optional).

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: A tuple containing the training dataset,
                                                          validation dataset, and test dataset.
        """
        # Step 1: Check if split files exist
        if not os.path.exists(self.split_path):
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
            'base_group': self.base_group,
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