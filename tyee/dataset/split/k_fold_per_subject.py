import os
import re
import logging
from copy import copy
from typing import List, Tuple, Union, Dict, Generator
import pandas as pd
from sklearn import model_selection
from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class KFoldPerSubject(BaseSplit):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the KFoldPerSubject class with the specified parameters.

        Args:
            n_splits (int): Number of folds for splitting.
            shuffle (bool): Whether to shuffle the data before splitting.
            random_state (Union[int, None]): Random seed for reproducibility.
            split_path (Union[None, str]): Path to save split files.
        """
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
        Create train and validation split files for each subject and fold.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        subjects = sorted(set(info['subject_id']))
        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            for fold_id, (train_index, val_index) in enumerate(self.k_fold.split(subject_info)):
                train_info = subject_info.iloc[train_index]
                val_info = subject_info.iloc[val_index]

                # Save train and validation splits to CSV files
                train_info.to_csv(
                    os.path.join(self.split_path, f'train_subject_{subject}_fold_{fold_id}.csv'),
                    index=False
                )
                val_info.to_csv(
                    os.path.join(self.split_path, f'val_subject_{subject}_fold_{fold_id}.csv'),
                    index=False
                )

    @property
    def subjects(self) -> List[str]:
        """
        Retrieve the list of unique subjects based on existing split files.

        Returns:
            List[str]: Sorted list of unique subject IDs.
        """
        indice_files = os.listdir(self.split_path)

        def indice_file_to_subject(indice_file: str) -> Union[str, None]:
            # Extract subject ID from file name using regex
            match = re.search(r'subject_(.*?)_fold_\d+.csv', indice_file)
            return match.group(1) if match else None

        # Filter and sort unique subject IDs
        subjects = list(set(map(indice_file_to_subject, indice_files)))
        return subjects

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
            match = re.search(r'subject_.*?_fold_(\d+).csv', indice_file)
            return int(match.group(1)) if match else None

        # Filter and sort unique fold IDs
        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        return fold_ids

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        subject: Union[int, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each subject and fold.

        Args:
            dataset (BaseDataset): The dataset to split.
            subject (Union[int, None]): Specific subject to split, or None for all subjects.

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
                    os.path.join(self.split_path, f'train_subject_{local_subject}_fold_{fold_id}.csv')
                )
                val_info = pd.read_csv(
                    os.path.join(self.split_path, f'val_subject_{local_subject}_fold_{fold_id}.csv')
                )

                # Create copies of the dataset for train and validation
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