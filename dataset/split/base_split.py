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

from dataset import BaseDataset
from typing import Generator, Tuple

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