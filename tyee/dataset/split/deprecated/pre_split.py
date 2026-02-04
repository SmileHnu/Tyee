#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : pre_split.py
@Time    : 2025/10/10 20:55:50
@Desc    : 
"""
import os
import re
import logging
from copy import copy
from typing import List, Tuple, Union, Dict, Generator
import pandas as pd
from sklearn import model_selection
from tyee.dataset.base_dataset import BaseDataset
from tyee.dataset.split.base_split import BaseSplit

log = logging.getLogger('split')

class PreSplitPerSubject(BaseSplit):
    def __init__(
        self,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the splitter.

        Args:
            split_path (Union[None, str]): Path to save/load the split files.
        """
        super().__init__(split_path=split_path, **kwargs)

    def split_info_constructor(
        self, 
        dataset: BaseDataset, 
        val_dataset: BaseDataset = None, 
        test_dataset: BaseDataset = None
    ) -> None:
        """
        Creates and saves split files for each subject based on pre-split datasets.
        
        It iterates through subjects found in the main dataset and saves their
        corresponding data from train, val, and test sets into separate CSV files.
        """
        subjects = sorted(dataset.info['subject_id'].unique())
        for subject in subjects:
            # Save train split
            train_info = dataset.info[dataset.info['subject_id'] == subject]
            train_info.to_csv(
                os.path.join(self.split_path, f'train_subject_{subject}.csv'),
                index=False
            )
            
            # Save validation split
            if val_dataset:
                val_info = val_dataset.info[val_dataset.info['subject_id'] == subject]
                val_info.to_csv(
                    os.path.join(self.split_path, f'val_subject_{subject}.csv'),
                    index=False
                )

            # Save test split
            if test_dataset:
                test_info = test_dataset.info[test_dataset.info['subject_id'] == subject]
                test_info.to_csv(
                    os.path.join(self.split_path, f'test_subject_{subject}.csv'),
                    index=False
                )

    @property
    def subjects(self) -> List[str]:
        """
        Retrieves the list of unique subjects based on existing split files.

        Returns:
            List[str]: A sorted list of unique subject IDs.
        """
        indice_files = os.listdir(self.split_path)

        def indice_file_to_subject(indice_file: str) -> Union[str, None]:
            match = re.search(r'subject_(.*?)\.csv', indice_file)
            return match.group(1) if match else None

        subjects = sorted(set(filter(None, map(indice_file_to_subject, indice_files))))
        return subjects

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
        subject: Union[int, str, None] = None
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Yields subject-specific datasets by loading pre-generated split files.

        If split files don't exist, it creates them first. Then, it loads the
        info for each subject and yields datasets with that specific info.
        """
        if not self.check_split_path():
            log.info('📊 | No existing split files found. Creating new ones.')
            log.info(
                f'😊 | Please set \033[92msplit_path\033[0m to \033[92m{self.split_path}\033[0m '
                'for the next run to reuse the same splits.'
            )
            os.makedirs(self.split_path, exist_ok=True)
            self.split_info_constructor(dataset, val_dataset, test_dataset)
        else:
            log.info(f'📊 | Detected existing split from {self.split_path}.')

        subjects = self.subjects
        if subject is not None:
            subject = str(subject)
            if subject not in subjects:
                raise ValueError(f'The subject {subject} is not in the available subject list: {subjects}.')
            subjects = [subject]

        for local_subject in subjects:
            # Load train dataset for the subject
            train_info_path = os.path.join(self.split_path, f'train_subject_{local_subject}.csv')
            train_info = pd.read_csv(train_info_path)
            train_subject_dataset = copy(dataset)
            train_subject_dataset.info = train_info
            
            # Load validation dataset for the subject
            val_subject_dataset = None
            val_info_path = os.path.join(self.split_path, f'val_subject_{local_subject}.csv')
            if val_dataset and os.path.exists(val_info_path):
                val_info = pd.read_csv(val_info_path)
                val_subject_dataset = copy(val_dataset)
                val_subject_dataset.info = val_info

            # Load test dataset for the subject
            test_subject_dataset = None
            test_info_path = os.path.join(self.split_path, f'test_subject_{local_subject}.csv')
            if test_dataset and os.path.exists(test_info_path):
                test_info = pd.read_csv(test_info_path)
                test_subject_dataset = copy(test_dataset)
                test_subject_dataset.info = test_info

            yield train_subject_dataset, val_subject_dataset, test_subject_dataset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split_path='{self.split_path}')"