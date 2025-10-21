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


class HoldOutPerSubjectGroupby(BaseSplit):
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
        Initialize the HoldOutPerSubjectGroupby class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'trial_id', 'session_id').
            val_size (float): Proportion of the dataset to include in the val split. Should be between 0.0 and 1.0. (default: 0.2)
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
        Create train and val split files for each subject.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        subjects = sorted(set(info['subject_id']))

        for subject in subjects:
            subject_info = info[info['subject_id'] == subject]

            train_infos = []
            val_infos = []

            group_ids = sorted(set(subject_info[self.group_by]))
            train_group_ids, val_group_ids = model_selection.train_test_split(
                group_ids,
                test_size=self.val_size,
                random_state=self.random_state,
                shuffle=self.shuffle
            )

            train_info = pd.concat(
                [subject_info[subject_info[self.group_by] == group_id] for group_id in train_group_ids],
                ignore_index=True
            )
            val_info = pd.concat(
                [subject_info[subject_info[self.group_by] == group_id] for group_id in val_group_ids],
                ignore_index=True
            )

            train_infos.append(train_info)
            val_infos.append(val_info)

            # Save train and val splits to CSV files
            combined_train_info = pd.concat(train_infos, ignore_index=True)
            combined_val_info = pd.concat(val_infos, ignore_index=True)

            combined_train_info.to_csv(
                os.path.join(self.split_path, f'train_subject_{subject}.csv'),
                index=False
            )
            combined_val_info.to_csv(
                os.path.join(self.split_path, f'val_subject_{subject}.csv'),
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

        def indice_file_to_subject(indice_file: str) -> str:
            match = re.search(r'subject_(.*?)\.csv', indice_file)
            return match.group(1) if match else None

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
        Generate train and val datasets for each subject.

        Args:
            dataset (BaseDataset): The dataset to split.
            subject (Union[int, None]): Specific subject to split, or None for all subjects.

        Yields:
            Tuple[BaseDataset, BaseDataset]: Train and val datasets for each subject.
        """
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

        subjects = self.subjects

        if subject is not None:
            subject = str(subject)
            assert subject in subjects, f'The subject should be in the subject list {subjects}.'

        for local_subject in subjects:
            if subject is not None and local_subject != subject:
                continue

            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_subject_{local_subject}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_subject_{local_subject}.csv')
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