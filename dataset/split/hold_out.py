import logging
import os
from copy import copy
from typing import Union, Tuple, Generator

import numpy as np
import pandas as pd
from sklearn import model_selection

from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class HoldOut(BaseSplit):
    def __init__(
        self,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initialize the HoldOut class with the specified parameters.

        Args:
            val_size (float): Proportion of the dataset to include in the val split. Should be between 0.0 and 1.0. (default: 0.2)
            shuffle (bool): Whether to shuffle the data before splitting. (default: False)
            random_state (Union[int, None]): Random seed for reproducibility. (default: None)
            split_path (Union[None, str]): Path to save split files. If None, a random path will be generated. (default: None)
        """
        self.val_size = val_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train and val split files.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information.
        """
        n_samples = len(info)
        indices = np.arange(n_samples)

        # Perform train-val split
        train_index, val_index = model_selection.train_test_split(
            indices,
            test_size=self.val_size,
            random_state=self.random_state,
            shuffle=self.shuffle
        )

        # Save train and val splits
        train_info = info.iloc[train_index]
        val_info = info.iloc[val_index]

        train_info.to_csv(os.path.join(self.split_path, 'train.csv'), index=False)
        val_info.to_csv(os.path.join(self.split_path, 'val.csv'), index=False)

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train and val datasets.

        Args:
            dataset (BaseDataset): The dataset to split.

        Returns:
            Tuple[BaseDataset, BaseDataset]: Train and val datasets.
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
            log.info(
                f'📊 | Detected existing split of train and val sets. Using existing split from {self.split_path}.'
            )
            log.info(
                '💡 | If the dataset is re-generated, you need to re-generate the split instead of using the previous split.'
            )

        # Load train and val splits
        train_info = pd.read_csv(os.path.join(self.split_path, 'train.csv'))
        val_info = pd.read_csv(os.path.join(self.split_path, 'val.csv'))

        # Create train and val datasets
        train_dataset = copy(dataset)
        train_dataset.info = train_info

        val_dataset = copy(dataset)
        val_dataset.info = val_info

        yield train_dataset, val_dataset, test_dataset

    @property
    def repr_body(self) -> dict:
        """
        Representation body for the class.

        Returns:
            dict: Dictionary containing class attributes.
        """
        return {
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