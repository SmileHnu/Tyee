#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : base_split.py
@Time    : 2025/04/09 10:29:16
@Desc    : 
"""

import os
import logging
from dataset import BaseDataset
from typing import Generator, Tuple, List

log = logging.getLogger('split')

class BaseSplit:
    def __init__(
        self,
        split_path: str,
        **kwargs
    ) -> None:
        """
        Initialize the BaseSplit class.

        Args:
            split_path (str): Path to the split dataset.
        """
        self.split_path = split_path
        
    def check_split_path(self) -> bool:
        """
        Check if the split path exists and contains at least one CSV file.

        Returns:
            bool: True if the split path exists and contains at least one CSV file, False otherwise.
        """
        if not os.path.exists(self.split_path):
            log.info(f"❌ | Split path '{self.split_path}' does not exist.")
            return False

        # Check if there are any .csv files in the directory
        csv_files = [f for f in os.listdir(self.split_path) if f.endswith('.csv')]
        if not csv_files:
            log.info(f"❌ | No CSV files found in '{self.split_path}'.")
            return False

        log.info(f"✅ | Found CSV files in '{self.split_path}': {csv_files}")
        return True

    def split(
        self, 
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Split the dataset into training and validation sets.
        Args:
            dataset (BaseDataset): The dataset to be split.
        Yields:
            Generator: A generator that yields the split datasets.
        """
        raise NotImplementedError("This method should be overridden by subclasses")