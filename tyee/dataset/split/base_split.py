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
from tyee.dataset import BaseDataset
from typing import Generator, Tuple, List

log = logging.getLogger('split')


class BaseSplit:
    def __init__(
        self,
        dst_path: str,
        **kwargs
    ) -> None:
        """
        Initialize the BaseSplit class.

        Args:
            dst_path (str): Path to the split dataset.
        """
        self.dst_path = dst_path
        
    def check_dst_path(self) -> bool:
        """
        Check if the dst path exists and contains at least one CSV file.

        Returns:
            bool: True if the dst path exists and contains at least one CSV file, False otherwise.
        """
        if not os.path.exists(self.dst_path):
            log.info(f"❌ | Split path '{self.dst_path}' does not exist.")
            return False

        # Check if there are any .csv files in the directory
        csv_files = [f for f in os.listdir(self.dst_path) if f.endswith('.csv')]
        if not csv_files:
            log.info(f"❌ | No CSV files found in '{self.dst_path}'.")
            return False

        log.info(f"✅ | Found CSV files in '{self.dst_path}': {csv_files}")
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
    
    def _to_number(self, rsize: List[float| int], total_size: int) -> List[int]:
        """
        Convert ratio-based rsize to number-based rsize.

        :param List[float| int] rsize: array of number or ratio of dataset split.
        :param int total_size: total number of samples in the dataset.
        :return: List[int]: array of number of dataset split.
        """
        if self.rtype == 'ratio':
            num_train = int(total_size * rsize[0])
            num_val = int(total_size * rsize[1])
            num_test = total_size - num_train - num_val
            return [num_train, num_val, num_test]
        elif self.rtype == 'number':
            return rsize
        else:
            raise ValueError(f"rtype should be either 'ratio' or 'number', but got {self.rtype}.")

    def _to_ratio(self, rsize: List[float| int]) -> List[float]:
        """
        Convert number-based rsize to ratio-based rsize.
        
        :param List[float| int] rsize: array of number of dataset split.
        :return: List[float]: array of ratio of dataset split.
        """
        if self.rtype == 'ratio':
            return self.rsize
        elif self.rtype == 'number':
            return [item / sum(rsize) for item in rsize]
        else:
            raise ValueError(f"rtype should be either 'ratio' or 'number', but got {self.rtype}.")
