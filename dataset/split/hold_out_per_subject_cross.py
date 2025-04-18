import logging
import os
import re
from copy import copy
from typing import Dict, List, Tuple, Union, Generator

import pandas as pd
from sklearn import model_selection

from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class HoldOutPerSubjectCross(BaseSplit):
    """
    A class for performing hold-out validation for each subject individually.

    This class splits the dataset into training and validation sets for each subject
    based on a specified grouping column (e.g., 'trial_id', 'session_id'). The split
    ensures that all rows belonging to the same group (defined by `group_by`) for a
    specific subject are either in the training set or the validation set, but not both.

    Attributes:
        group_by (str): The column name used for grouping within each subject (e.g., 'trial_id').
        val_size (float): The proportion of the dataset to include in the validation split.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (Union[int, None]): Random seed for reproducibility.
        split_path (Union[None, str]): Path to save the split files.
    """

    def __init__(
        self,
        group_by: str,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOutPerSubjectCross class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id').
            val_size (float): Proportion of the dataset to include in the validation split. Should be between 0.0 and 1.0. (default: 0.2)
            shuffle (bool): Whether to shuffle the data before splitting. (default: False)
            random_state (Union[int, None]): Random seed for reproducibility. (default: None)
            split_path (Union[None, str]): Path to save split files. (default: None)
        """
        self.group_by = group_by
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each subject.

        This method performs the following steps:
        1. Extract all unique subjects from the `subject_id` column.
        2. For each subject:
           - Extract all unique groups (`group_by`) within the subject.
           - Perform a train-validation split on the groups.
           - Assign rows belonging to the training and validation groups to separate DataFrames.
        3. Save the training and validation DataFrames for each subject as separate CSV files.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information. This DataFrame
                                 must include the columns `subject_id` and `group_by`.
        """
        # Step 1: Extract all unique subjects
        subjects = list(set(info['subject_id']))

        # Step 2: Process each subject
        for subject in subjects:
            # Filter rows belonging to the current subject
            subject_info = info[info['subject_id'] == subject]

            # Extract unique groups within the subject
            group_ids = list(set(subject_info[self.group_by]))

            # Perform train-validation split on the groups
            train_group_ids, val_group_ids = model_selection.train_test_split(
                group_ids,
                test_size=self.val_size,
                random_state=self.random_state,
                shuffle=self.shuffle
            )

            # Assign rows to training and validation sets based on the split group IDs
            train_info = pd.concat(
                [subject_info[subject_info[self.group_by] == group_id] for group_id in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [subject_info[subject_info[self.group_by] == group_id] for group_id in val_group_ids],
                ignore_index=True
            )

            # Step 3: Save train and validation splits to CSV files
            train_info.to_csv(
                os.path.join(self.split_path, f'train_subject_{subject}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_subject_{subject}.csv'),
                index=False
            )

    @property
    def subjects(self) -> List[str]:
        """
        Retrieve the list of unique subjects based on existing split files.

        This method scans the `split_path` directory for files matching the pattern
        `train_subject_<subject_id>.csv` or `val_subject_<subject_id>.csv` and extracts
        the unique subject IDs.

        Returns:
            List[str]: Sorted list of unique subject IDs.
        """
        indice_files = os.listdir(self.split_path)

        def indice_file_to_subject(indice_file: str) -> Union[str, None]:
            # Extract subject ID from file name using regex
            match = re.search(r'subject_(.*?)\.csv', indice_file)
            return match.group(1) if match else None

        # Filter and sort unique subject IDs
        subjects = sorted(set(filter(None, map(indice_file_to_subject, indice_files))))
        return subjects

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        subject: Union[int, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each subject.

        This method performs the following steps:
        1. Check if the split files exist in the specified `split_path`.
           - If not, call `split_info_constructor` to create them.
        2. Load the split files for each subject into DataFrames.
        3. Create shallow copies of the input dataset and assign the loaded DataFrames
           to their `info` attributes.
        4. Yield the training dataset, validation dataset, and test dataset for each subject.

        Args:
            dataset (BaseDataset): The dataset to split.
            subject (Union[int, None]): Specific subject to split, or None for all subjects.

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: Train, validation, and test datasets for each subject.
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

        # Retrieve the list of subjects
        subjects = self.subjects

        # Validate the specified subject
        if subject is not None:
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        # Step 2: Process each subject
        for local_subject in subjects:
            if subject is not None and local_subject != subject:
                continue

            # Load train and validation splits for the current subject
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_subject_{local_subject}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_subject_{local_subject}.csv')
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