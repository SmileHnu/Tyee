import logging
import os
import re
from copy import copy
from typing import List, Tuple, Union, Generator
import pandas as pd

from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit


log = logging.getLogger('split')


class LeaveOneOut(BaseSplit):
    def __init__(
        self, 
        group_by: str,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the LeaveOneOut class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'subject_id', 'session_id', 'trial_id').
            split_path (Union[None, str]): Path to save split files.
        """
        self.group_by = group_by
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each group.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        groups = sorted(set(info[self.group_by]))

        for val_group in groups:
            train_groups = groups.copy()
            train_groups.remove(val_group)

            train_info = []
            for train_group in train_groups:
                train_info.append(info[info[self.group_by] == train_group])

            train_info = pd.concat(train_info, ignore_index=True)
            val_info = info[info[self.group_by] == val_group]

            train_info.to_csv(
                os.path.join(self.split_path, f'train_{self.group_by}_{val_group}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_{self.group_by}_{val_group}.csv'),
                index=False
            )

    @property
    def groups(self) -> List:
        """
        Retrieve the list of unique groups based on existing split files.

        Returns:
            List: Sorted list of unique group IDs.
        """
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_group(indice_file):
            # Extract group ID dynamically based on `group_by`
            pattern = rf'{self.group_by}_(.*).csv'
            match = re.search(pattern, indice_file)
            return match.group(1) if match else None

        groups = list(set(map(indice_file_to_group, indice_files)))
        groups.sort()
        return groups

    def split(
        self, 
        dataset: BaseDataset, 
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        group: Union[str, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each group.

        Args:
            dataset (BaseDataset): The dataset to split.
            group (Union[str, None]): Specific group to split, or None for all groups.

        Yields:
            Tuple[BaseDataset, BaseDataset]: Train and validation datasets for each group.
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
            log.info(
                f'📊 | Detected existing split of train and validation sets. Using existing split from {self.split_path}.'
            )
            log.info(
                '💡 | If the dataset is re-generated, you need to re-generate the split instead of using the previous split.'
            )

        groups = self.groups
        if group is not None:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_{self.group_by}_{group}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_{self.group_by}_{group}.csv')
            )
            train_dataset = copy(dataset)
            train_dataset.info = train_info

            val_dataset = copy(dataset)
            val_dataset.info = val_info

            yield train_dataset, val_dataset, test_dataset
        else:
            for group in groups:
                train_info = pd.read_csv(
                    os.path.join(self.split_path, f'train_{self.group_by}_{group}.csv')
                )
                val_info = pd.read_csv(
                    os.path.join(self.split_path, f'val_{self.group_by}_{group}.csv')
                )

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                val_dataset = copy(dataset)
                val_dataset.info = val_info

                yield train_dataset, val_dataset, test_dataset


class LeaveOneOutEAndT(BaseSplit):
    def __init__(
        self, 
        group_by: str,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the LeaveOneOut class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'subject_id', 'session_id', 'trial_id').
            split_path (Union[None, str]): Path to save split files.
        """
        self.group_by = group_by
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each group.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        groups = sorted(set(info[self.group_by]))

        for idx, test_group in enumerate(groups):
            val_group = groups[(idx + len(groups) - 1) % len(groups)]
            train_groups = groups.copy()
            train_groups.remove(test_group)
            train_groups.remove(val_group)

            train_info = []
            for train_group in train_groups:
                train_info.append(info[info[self.group_by] == train_group])

            train_info = pd.concat(train_info, ignore_index=True)
            val_info = info[info[self.group_by] == val_group]
            test_info = info[info[self.group_by] == test_group]

            train_info.to_csv(
                os.path.join(self.split_path, f'train_{self.group_by}_{test_group}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_{self.group_by}_{test_group}.csv'),
                index=False
            )
            test_info.to_csv(
                os.path.join(self.split_path, f'test_{self.group_by}_{test_group}.csv'),
                index=False
            )

    @property
    def groups(self) -> List:
        """
        Retrieve the list of unique groups based on existing split files.

        Returns:
            List: Sorted list of unique group IDs.
        """
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_group(indice_file):
            # Extract group ID dynamically based on `group_by`
            pattern = rf'{self.group_by}_(.*).csv'
            match = re.search(pattern, indice_file)
            return match.group(1) if match else None

        groups = list(set(map(indice_file_to_group, indice_files)))
        groups.sort()
        return groups

    def split(
        self, 
        dataset: BaseDataset, 
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        group: Union[str, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each group.

        Args:
            dataset (BaseDataset): The dataset to split.
            group (Union[str, None]): Specific group to split, or None for all groups.

        Yields:
            Tuple[BaseDataset, BaseDataset]: Train and validation datasets for each group.
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
            log.info(
                f'📊 | Detected existing split of train and validation sets. Using existing split from {self.split_path}.'
            )
            log.info(
                '💡 | If the dataset is re-generated, you need to re-generate the split instead of using the previous split.'
            )

        groups = self.groups
        if group is not None:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_{self.group_by}_{group}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_{self.group_by}_{group}.csv')
            )
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_{self.group_by}_{group}.csv')
            )
            train_dataset = copy(dataset)
            train_dataset.info = train_info

            val_dataset = copy(dataset)
            val_dataset.info = val_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, val_dataset, test_dataset
        else:
            for group in groups:
                train_info = pd.read_csv(
                    os.path.join(self.split_path, f'train_{self.group_by}_{group}.csv')
                )
                val_info = pd.read_csv(
                    os.path.join(self.split_path, f'val_{self.group_by}_{group}.csv')
                )
                test_info = pd.read_csv(
                    os.path.join(self.split_path, f'test_{self.group_by}_{group}.csv')
                )
                train_dataset = copy(dataset)
                train_dataset.info = train_info

                val_dataset = copy(dataset)
                val_dataset.info = val_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, val_dataset, test_dataset


from tyee.dataset.split.train_test_split import train_test_split
class LeaveOneOutAndHoldOutET(BaseSplit):
    def __init__(
        self, 
        group_by: str,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        stratify: str = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the LeaveOneOut class with the specified parameters.

        Args:
            group_by (str): The column name to group by (e.g., 'subject_id', 'session_id', 'trial_id').
            split_path (Union[None, str]): Path to save split files.
        """
        self.group_by = group_by
        self.file_group = group_by.split('_')[0]
        self.split_path = split_path
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
    
    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and validation split files for each group.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        groups = sorted(set(info[self.group_by]))

        for val_group in groups:
            train_groups = groups.copy()
            train_groups.remove(val_group)

            train_info = []
            for train_group in train_groups:
                train_info.append(info[info[self.group_by] == train_group])

            train_info = pd.concat(train_info, ignore_index=True)
            val_info = info[info[self.group_by] == val_group]
            val_info, test_info = train_test_split(
                val_info,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=self.stratify
            )
            train_info.to_csv(
                os.path.join(self.split_path, f'train_{self.file_group}_{val_group}.csv'),
                index=False
            )
            val_info.to_csv(
                os.path.join(self.split_path, f'val_{self.file_group}_{val_group}.csv'),
                index=False
            )
            test_info.to_csv(
                os.path.join(self.split_path, f'test_{self.file_group}_{val_group}.csv'),
                index=False
            )

    @property
    def groups(self) -> List:
        """
        Retrieve the list of unique groups based on existing split files.

        Returns:
            List: Sorted list of unique group IDs.
        """
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_group(indice_file):
            # Extract group ID dynamically based on `group_by`
            pattern = rf'{self.file_group}_(.*).csv'
            match = re.search(pattern, indice_file)
            return match.group(1) if match else None

        groups = list(set(map(indice_file_to_group, indice_files)))
        groups.sort()
        return groups

    def split(
        self, 
        dataset: BaseDataset, 
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        group: Union[str, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and validation datasets for each group.

        Args:
            dataset (BaseDataset): The dataset to split.
            group (Union[str, None]): Specific group to split, or None for all groups.

        Yields:
            Tuple[BaseDataset, BaseDataset]: Train and validation datasets for each group.
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
            log.info(
                f'📊 | Detected existing split of train and validation sets. Using existing split from {self.split_path}.'
            )
            log.info(
                '💡 | If the dataset is re-generated, you need to re-generate the split instead of using the previous split.'
            )

        groups = self.groups
        if group is not None:
            train_info = pd.read_csv(
                os.path.join(self.split_path, f'train_{self.file_group}_{group}.csv')
            )
            val_info = pd.read_csv(
                os.path.join(self.split_path, f'val_{self.file_group}_{group}.csv')
            )
            test_info = pd.read_csv(
                os.path.join(self.split_path, f'test_{self.file_group}_{group}.csv')
            )
            train_dataset = copy(dataset)
            train_dataset.info = train_info

            val_dataset = copy(dataset)
            val_dataset.info = val_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            yield train_dataset, val_dataset, test_dataset
        else:
            for group in groups:
                train_info = pd.read_csv(
                    os.path.join(self.split_path, f'train_{self.file_group}_{group}.csv')
                )
                val_info = pd.read_csv(
                    os.path.join(self.split_path, f'val_{self.file_group}_{group}.csv')
                )
                test_info = pd.read_csv(
                    os.path.join(self.split_path, f'test_{self.file_group}_{group}.csv')
                )

                train_dataset = copy(dataset)
                train_dataset.info = train_info

                val_dataset = copy(dataset)
                val_dataset.info = val_info

                test_dataset = copy(dataset)
                test_dataset.info = test_info

                yield train_dataset, val_dataset, test_dataset