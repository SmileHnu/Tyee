#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : base_dataset.py
@Time    : 2025/02/21 16:52:17
@Desc    : 
"""

import logging
import os
import shutil
import ast
from copy import copy, deepcopy
from collections import defaultdict, OrderedDict
from typing import Any, Callable, Dict, Union, List, Tuple
from sklearn.model_selection import KFold, train_test_split

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset.io import PhysioSignalIO
from dataset.io import MetaInfoIO


log = logging.getLogger('dataset')

def merge_info(info_merged: List[Dict]) -> Dict[str, List]:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        info_merged (List[Dict]): List of dictionaries to merge.

    Returns:
        Dict[List]: Merged dictionary with lists as values.
    """
    merged_info = defaultdict(list)
    for info in info_merged:
        for key, value in info.items():
            merged_info[key].append(value)
    merged_info = {k: pd.concat(v, ignore_index=True) for k,v in merged_info.items()}
    return merged_info

class BaseDataset(Dataset):
    """
    BaseDataset class for managing physiological signal datasets.

    This class provides functionalities for:
    - Initializing and processing datasets.
    - Handling lazy and eager loading modes.
    - Reading and writing signal data.
    - Managing metadata and post-processing hooks.
    """

    def __init__(
        self,
        root_path: str,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        include_end: bool = False,
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        lazy_threshold: int = 128,
        verbose: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize the BaseDataset.

        Args:
            io_path (str): Path to the dataset storage.
            io_mode (str): Storage mode ('lmdb', 'memory', etc.).
            io_size (int): Size of the storage.
            lazy_threshold (int): Threshold for switching to lazy loading.
            num_worker (int): Number of workers for parallel processing.
            verbose (bool): Whether to display progress bars.
            after_trial (Callable): Hook for processing after each trial.
            after_session (Callable): Hook for processing after each session.
            after_subject (Callable): Hook for processing after each subject.
            signal_types (List[str]): List of signal types to process.
            **kwargs: Additional arguments for dataset processing.
        """
        self.root_path = root_path
        self.before_segment_transform = before_segment_transform
        self.offline_signal_transform = offline_signal_transform
        self.offline_label_transform = offline_label_transform
        self.online_signal_transform = online_signal_transform
        self.online_label_transform = online_label_transform
        self.io_path = io_path
        self.io_size = io_size
        self.io_chunks = io_chunks
        self.io_mode = io_mode
        self.lazy_threshold = lazy_threshold
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.include_end = include_end
        self.num_worker = num_worker
        self.verbose = verbose
        
        # Check if the dataset folder is empty or in memory mode
        if self.is_folder_empty(self.io_path) or self.io_mode == 'memory':
            log.info(
                f'🔍 | No cached processing results found, processing data from {self.io_path}.')
            os.makedirs(self.io_path, exist_ok=True)

            records = self.set_records(self.root_path, **kwargs)
            if self.num_worker == 0:
                try:
                    worker_results = []
                    for record_id, record in tqdm(enumerate(records),
                                                  disable=not self.verbose,
                                                  desc="[PROCESS]",
                                                  total=len(records),
                                                  position=0,
                                                  leave=None):
                        worker_results.append(
                            self.handle_record(
                                               record=record,
                                               record_id=record_id,
                                               **kwargs))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
            else:
                # catch the exception
                try:
                    worker_results = Parallel(n_jobs=self.num_worker)(
                        delayed(self.handle_record)(
                            record_id=record_id,
                            record=record,
                            **kwargs)
                        for record_id, record in tqdm(enumerate(records),
                                                      disable=not self.verbose,
                                                      desc="[PROCESS]",
                                                      total=len(records),
                                                      position=0,
                                                      leave=None))
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e

            if not self.io_mode == 'memory':
                log.info(
                    f'✅ | All processed data has been cached to {io_path}.'
                )
                log.info(
                    f'😊 | Please set \033[92mio_path\033[0m to \033[92m{io_path}\033[0m for the next run, to directly read from the cache if you wish to skip the data processing step.'
                )

            self.init_io(
                io_path=self.io_path,
                io_size=self.io_size,
                io_chunks=self.io_chunks,
                io_mode=self.io_mode)  

        else:
            log.info(
                f'🔍 | Detected cached processing results, reading cache from {self.io_path}.'
            )
            self.init_io(
                io_path=self.io_path,
                io_size=self.io_size,
                io_chunks=self.io_chunks,
                io_mode=self.io_mode)

    def is_folder_empty(self, folder_path: str) -> bool:
        """
        Check if the folder structure is valid and contains the required data.

        Args:
            folder_path (str): Path to the top-level folder to check.

        Returns:
            bool: True if the folder is empty, missing required files, or contains only empty files. False otherwise.
        """
        if not os.path.exists(folder_path):
            return True

        # Check for record folders
        record_folders = [f for f in os.listdir(folder_path) if f.startswith("record_") and os.path.isdir(os.path.join(folder_path, f))]
        if not record_folders:
            return True

        has_valid_record = False

        # Validate each record folder
        for record_folder in record_folders:
            record_path = os.path.join(folder_path, record_folder)

            # Check for info.csv
            metadata_file_path = os.path.join(record_path, "info.csv")
            if not os.path.exists(metadata_file_path) or os.path.getsize(metadata_file_path) == 0:
                log.warning(f"Missing or empty info.csv in {record_path}. Deleting this record.")
                shutil.rmtree(record_path)
                continue

            # Check for non-empty signal folders
            signal_folders = [f for f in os.listdir(record_path) if os.path.isdir(os.path.join(record_path, f)) and f != "info.csv"]
            has_non_empty_signal = False
            for signal_folder in signal_folders:
                signal_folder_path = os.path.join(record_path, signal_folder)
                for root, dirs, files in os.walk(signal_folder_path):
                    if any(os.path.getsize(os.path.join(root, file)) > 0 for file in files):
                        has_non_empty_signal = True
                        break
                if has_non_empty_signal:
                    break

            if not has_non_empty_signal:
                log.warning(f"No valid signal data found in {record_path}. Deleting this record.")
                shutil.rmtree(record_path) 
                continue

            has_valid_record = True

        return not has_valid_record

    def init_io(self, io_path: str, io_size: int, io_chunks: int, io_mode: str):
        """
        Initialize IO for the dataset.

        Args:
            io_path (str): Path to the dataset storage.
            io_size (int): Size of the storage.
            io_mode (str): Storage mode ('lmdb', 'memory', 'pickle').
        """
        # get all records
        records = os.listdir(io_path)
        # filter the records with the prefix 'record_'
        records = list(filter(lambda x: 'record_' in x, records))
        # sort the records
        records = sorted(records, key=lambda x: int(x.split('_')[1]))

        assert len(records) > 0, \
            f"The io_path, {io_path}, is corrupted. Please delete this folder and try again."

        info_merged = []
        signals_info_merged = []
        labels_info_merged = []
        self.signal_cache = {}
        self.label_cache = {}
        if len(records) > self.lazy_threshold:
            # Store paths instead of PhysioSignalIO instances
            self.signal_paths = {}
            self.label_paths = {}
            for record in records:
                meta_info_io_path = os.path.join(io_path, record, 'info.csv')
                info_io = MetaInfoIO(meta_info_io_path)
                info_df = info_io.read_all()
                # check DataFrame is empty
                if info_df.empty:
                    log.warning(f"Empty DataFrame for record: {record}, skipping.")
                    continue
                
                assert 'record_id' not in info_df.columns, \
                    "column 'record_id' is a forbidden reserved word."
                info_df['record_id'] = record
                info_merged.append(info_df)

                self.signal_paths[record] = {}
                signal_path = os.path.join(io_path, record, 'signals')
                self.signal_paths[record] = signal_path
                signal_info_df = {}
                for signal_type in os.listdir(signal_path):
                    signal_info_path = os.path.join(signal_path, signal_type, 'info.csv')
                    signal_info_io = MetaInfoIO(signal_info_path)
                    signal_info_df[signal_type] = signal_info_io.read_all()
                    signals_info_merged.append(signal_info_df)
                    


                label_path = os.path.join(io_path, record, 'labels')
                self.label_paths[record] = label_path
                label_info_df = {}
                for label_type in os.listdir(label_path):
                    label_info_path = os.path.join(label_path, label_type, 'info.csv')
                    label_info_io = MetaInfoIO(label_info_path)
                    label_info_df[label_type] = label_info_io.read_all()
                    labels_info_merged.append(label_info_df)


            self.signal_io_router = {}  # Not used in lazy mode
            self.label_io_router = {}
        else:
            self.signal_io_router = {}
            self.label_io_router = {}
            for record in records:
                meta_info_io_path = os.path.join(io_path, record, 'info.csv')
                info_io = MetaInfoIO(meta_info_io_path)
                info_df = info_io.read_all()
                if info_df.empty:
                    log.warning(f"Empty DataFrame for record: {record}, skipping.")
                    continue
                
                assert 'record_id' not in info_df.columns, \
                    "column 'record_id' is a forbidden reserved word."
                info_df['record_id'] = record
                info_merged.append(info_df)

                self.signal_io_router[record] = {}
                signal_io_path = os.path.join(io_path, record, 'signals')
                signal_io = PhysioSignalIO(signal_io_path,
                                        io_size=io_size,
                                        io_mode=io_mode,
                                        io_chunks=io_chunks)
                self.signal_io_router[record] = signal_io
                signal_info_df = {}
                for signal_type in os.listdir(signal_io_path):
                    signal_info_path = os.path.join(signal_io_path, signal_type, 'info.csv')
                    signal_info_io = MetaInfoIO(signal_info_path)
                    signal_info_df[signal_type] = signal_info_io.read_all()
                    signals_info_merged.append(signal_info_df)


                self.label_io_router[record] = {}
                label_io_path = os.path.join(io_path, record, 'labels')
                label_io = PhysioSignalIO(label_io_path,
                                        io_size=io_size,
                                        io_mode=io_mode,
                                        io_chunks=io_chunks)
                self.label_io_router[record] = label_io
                label_info_df = {}
                for label_type in os.listdir(label_io_path):
                    label_info_path = os.path.join(label_io_path, label_type, 'info.csv')
                    label_info_io = MetaInfoIO(label_info_path)
                    label_info_df[label_type] = label_info_io.read_all()
                    labels_info_merged.append(label_info_df)


            self.signal_paths = {}  # Not used in eager mode
            self.label_paths = {}

        self.info = pd.concat(info_merged, ignore_index=True)
        self.signals_info = merge_info(signals_info_merged)
        self.labels_info = merge_info(labels_info_merged)

    def is_lazy(self) -> bool:
        """
        Initialize IO for the dataset.

        Args:
            io_path (str): Path to the dataset storage.
            io_size (int): Size of the storage.
            io_mode (str): Storage mode ('lmdb', 'memory', etc.).
        """
        assert hasattr(self, 'signal_io_router') or hasattr(
            self, 'signal_paths'), "The dataset should contain signal_io_router or signal_paths."
        if hasattr(self, 'signal_io_router') and self.signal_io_router is not None and len(self.signal_io_router) > 0:
            return False
        if hasattr(self, 'signal_paths') and self.signal_paths is not None and len(self.signal_paths) > 0:
            return True
        raise ValueError("Both signal_io_router and signal_paths are empty.")

    def write_signal(self, record: str, key: str, signals: dict):
        """
        Write a signal to the dataset.

        Args:
            record (str): The record identifier.
            key (str): The key of the signal to write.
            signal (Any): The signal data to write.
            signal_type (str): The type of signal to write.
        """
        for signal_type, signal in signals.items():
            info = deepcopy(signal['info'])
            del signal['info']
            # write signal data
            signal_path = os.path.join(self.io_path, record, 'signals')
            signal_io = PhysioSignalIO(
                signal_path, 
                io_size=self.io_size, 
                io_mode=self.io_mode, 
                io_chunks=self.io_chunks)
            signal_io.write_signal(signal, signal_type, key)
            # write info
            info_path = os.path.join(signal_path, signal_type, 'info.csv')
            info_io = MetaInfoIO(info_path)
            for sample_id, win in zip(info['sample_ids'], info['windows']):
                info_row = {
                    'sample_id': sample_id,
                    'segment_id': key,
                    'start': win['start'],
                    'end': win['end'],
                }
                info_io.write_info(info_row)

    def write_label(self, record: str, key: str, labels: dict):
        for label_type, label in labels.items():
            info = deepcopy(label['info'])
            del label['info']
            # write signal data
            signal_path = os.path.join(self.io_path, record, 'labels')
            signal_io = PhysioSignalIO(
                signal_path, 
                io_size=self.io_size, 
                io_mode=self.io_mode,
                io_chunks=self.io_chunks
                )
            signal_io.write_signal(label, label_type, key)
            # write info
            info_path = os.path.join(signal_path, label_type, 'info.csv')
            info_io = MetaInfoIO(info_path)
            if 'windows' not in info:
                for sample_id in info['sample_ids']:
                    info_row = {
                        'sample_id': sample_id,
                        'segment_id': key,
                    }
                    info_io.write_info(info_row)
            else:
                for sample_id, win in zip(info['sample_ids'], info['windows']):
                    info_row = {
                        'sample_id': sample_id,
                        'segment_id': key,
                        'start': win['start'],
                        'end': win['end'],
                    }
                    info_io.write_info(info_row)
    
    def write_info(self, record: str, info: dict):
        """
        Write metadata information to MetaInfoIO.

        Args:
            record (str): The record identifier.
            key (str): The key of the signal to write.
            info (dict): The metadata information to write.
        """
        info_path = os.path.join(self.io_path, record, 'info.csv')
        info_io = MetaInfoIO(info_path)
        sample_ids = deepcopy(info['sample_ids'])
        del info['sample_ids']
        for sample_id in sample_ids:
            info_row = {
                'sample_id': sample_id
            }
            info_row.update(info)
            info_io.write_info(info_row)

    def read_signal(self, record: str, sample_id: str) -> dict:
        """
        Read a signal from the dataset.

        Args:
            record (str): The record identifier.
            key (str): The key of the signal to read.
            signal_type (str): The type of signal to read.

        Returns:
            Any: The signal data.
        """
        if self.is_lazy():
            # Create temporary PhysioSignalIO instance
            signal_io = PhysioSignalIO(
                self.signal_paths[record],
                io_size=self.io_size,
                io_mode=self.io_mode,
                io_chunks=self.io_chunks
            )
        else:
            signal_io = self.signal_io_router[record]
        signals = {}
        for signal_type in signal_io.signal_types():
            df = self.signals_info[signal_type]
            info = df[df['sample_id'] == sample_id].iloc[0].to_dict()
            key = str(info['segment_id'])
            start = int(info['start'])
            end = int(info['end'])
            signal = signal_io.read_signal(signal_type, key, start, end)
            signals[signal_type] = signal
            
        return signals
    
    def read_label(self, record: str, sample_id: str) -> dict:
        """
        Read a label from the dataset.

        Args:
            record (str): The record identifier.
            key (str): The key of the signal to read.
            signal_type (str): The type of signal to read.

        Returns:
            Any: The signal data.
        """
        if self.is_lazy():
            # Create temporary PhysioSignalIO instance
            label_io = PhysioSignalIO(
                self.label_paths[record],
                io_size=self.io_size,
                io_mode=self.io_mode,
                io_chunks=self.io_chunks
            )
        else:
            label_io = self.label_io_router[record]
        labels = {}
        for label_type in label_io.signal_types():
            df = self.labels_info[label_type]
            info = df[df['sample_id'] == sample_id].iloc[0].to_dict()
            key = str(info['segment_id'])
            start = info.get('start', None)
            end = info.get('end', None)
            label = label_io.read_signal(label_type, key, start, end)
            
            labels[label_type] = copy(label)
                
        return labels
        
    def read_info(self, index: int) -> Dict:
        """
        Retrieve metadata information from MetaInfoIO based on the given index.

        The metadata must include `clip_id`, which specifies the corresponding key
        in PhusioSignalIO. This key can be used to index samples via `self.read_signal(key)`.

        Args:
            index (int): The index of the metadata to retrieve.

        Returns:
            dict: The metadata information as a dictionary.
        """
        info = self.info.iloc[index].to_dict()
        return info

    def exist(self, io_path: str) -> bool:
        """
        Check if the database IO exists.

        Args:
            io_path (str): Path to the database IO.

        Returns:
            bool: True if the database IO exists, False otherwise.
        """

        return os.path.exists(io_path)

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieve a dataset item by index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing signal data and labels.
        """
        info = self.read_info(index)
        sample_id = str(info['sample_id'])
        # print(sample_id)
        record = str(info['record_id'])
        # print(record, sample_id)
        signals = self.read_signal(record, sample_id)
        signals = self.apply_transform(self.online_signal_transform, signals)

        labels = self.read_label(record, sample_id)
        labels = self.apply_transform(self.online_label_transform, labels)

        return self.assemble_sample(signals,labels)

    def get_labels(self) -> list:
        """
        Retrieve all labels from the dataset.

        Returns:
            list: A list of labels from the dataset.
        """
        labels = []
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels

    def __len__(self) -> int:
        """
        Get the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.info)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching data.

        This function processes a batch of data, which is a list of dictionaries,
        and combines them into a single dictionary with batched tensors or arrays.

        Args:
            batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                        represents a single data sample.

        Returns:
            Dict[str, Any]: A dictionary containing batched data.
        """
        # Initialize a dictionary to store batched signals
        batch_signals = {key: [] for key in batch[0]}

        # Collect data for each key
        for item in batch:
            for key, value in item.items():
                batch_signals[key].append(value)

        # Recursive function to process nested dictionaries
        def recursive_collate(data: List[Any]) -> Any:
            """
            Recursively collate data into tensors or arrays.

            Args:
                data (List[Any]): A list of data items to be collated.

            Returns:
                Any: Collated data as tensors, arrays, or other supported types.
            """
            # print('collate:', type(data[0]), data[0])
            if isinstance(data[0], dict):
                # Process nested dictionaries
                return {key: recursive_collate([d[key] for d in data]) for key in data[0]}
            elif isinstance(data[0], torch.Tensor):
                # Stack tensors
                return torch.stack(data)
            elif isinstance(data[0], np.ndarray):
                # Convert numpy arrays to tensors
                return torch.tensor(np.stack(data))
            elif isinstance(data[0], (int, float, list, np.integer)):
                # Convert scalars or lists to tensors
                return torch.tensor(data)
            else:
                # Return data as-is for unsupported types
                return data

        # Apply recursive collation to each key
        for key in batch_signals:
            batch_signals[key] = recursive_collate(batch_signals[key])

        return batch_signals  

    @staticmethod
    def get_subject_id(**kwargs) -> str:
        """
        Retrieve the subject ID for a specific record.

        Args:
            **kwargs: Additional parameters.

        Returns:
            str: The subject ID. Default is '0'.
        """
        return "0"

    @staticmethod
    def get_session_id(**kwargs) -> str:
        """
        Retrieve the session ID for a specific record.

        Args:
            **kwargs: Additional parameters.

        Returns:
            str: The session ID. Default is '0'.
        """
        return "0"

    @staticmethod
    def get_trial_id(**kwargs) -> str:
        """
        Retrieve the trial ID for a specific record.

        Args:
            **kwargs: Additional parameters.

        Returns:
            str: The trial ID. Default is '0'.
        """
        return "0"

    def set_records(self, **kwargs) -> List:
        """
        Generate database blocks for processing.

        This method is used to describe which data blocks need to be processed
        to generate the database. It is called in the `__init__` method of the class
        and executed in parallel using `joblib.Parallel`.

        Args:
            **kwargs: Parameters derived from the class `__init__`.

        Example:
            def set_records(self, root_path: str = None, **kwargs):
                # Return a list of filenames to be processed by `process_record`
                return os.listdir(root_path)

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Method set_records is not implemented in class BaseDataset")

    def handle_record(
        self,
        record: Any = None,
        record_id: Union[int, str] = None,
        **kwargs
    ) -> Dict:
        """
        Handle a single record for database generation.

        This method processes a single record and writes the processed data to the database.

        Args:
            io_path (str): Path to the database storage.
            io_size (int): Size of the storage.
            io_mode (str): Storage mode ('lmdb', 'memory', etc.).
            record (Any): The record to process.
            record_id (Union[int, str]): The identifier of the record.
            read_record (Callable): Function to read the record.
            process_record (Callable): Function to process the record.
            signal_types (list): List of signal types to process.
            verbose (bool): Whether to display progress bars.
            **kwargs: Additional arguments for processing.

        Returns:
            Dict: A dictionary containing the record identifier.
        """
        _record_id = f'record_{str(record_id)}'

        kwargs['record'] = record
        kwargs.update(self.read_record(**kwargs))
        gen = self.process_record(**kwargs)

        if record_id == 0:
            pbar = tqdm(disable=not self.verbose,
                        desc=f"[RECORD {record}]",
                        position=1,
                        leave=None)

        while True:
            try:
                obj = next(gen)
                if record_id == 0:
                    pbar.update(1)
            except StopIteration:
                break

            if obj and 'key' in obj:
                if 'signals' in obj:
                    # write signal
                    signals = obj['signals']
                    self.write_signal(_record_id, obj['key'], signals)
                if 'labels' in obj:
                    # write label
                    labels = obj['labels']
                    self.write_label(_record_id, obj['key'], labels)
                if 'info' in obj:
                    # write info
                    info = obj['info']
                    self.write_info(_record_id, info)

        if record_id == 0:
            pbar.close()

        return {
            'record': _record_id
        }
    
    def read_record(self, record: str | tuple, **kwargs) -> Dict:
        """
        Read a record from the database.

        This method describes how to read a file. It is called in the `__init__` method
        of the class and executed in parallel using `joblib.Parallel`. The output of this
        method is passed to `process_record`.

        Args:
            record (Any): The record to process. It is an element from the list returned by `set_records`.
            **kwargs: Parameters derived from the class `__init__`.

        Example:
            def read_record(**kwargs):
                return {
                    'eeg':{
                        'signal': eeg_signal,
                        'channels': eeg_channels,
                        'freq': eeg_freq,  
                    },
                    'ecg':{
                        'signal': ecg_signal,
                        'channels': ecg_channels,
                        'freq': ecg_freq,
                    },
                    ...,
                    'labels':{
                        'segments':[{
                            'start': 0.5,
                            'end': 2.3, 
                            'value': {
                                'label':{
                                    'data': 1,
                                }
                            }
                        }]
                        
                        ...,
                    },
                    'meta':{
                        'filename': filename
                        ...

                    }
                }

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """

        raise NotImplementedError(
            "Method read_record is not implemented in class BaseDataset"
        )

    def process_record(self, result: Dict, **kwargs) -> Dict:
        """
        Process a record to generate the database.

        This method describes how to process a file to generate the database. It is called
        in the `__init__` method of the class and executed in parallel using `joblib.Parallel`.

        Args:
            result (Any): The result to process. It is an Dict returned by `read_records`.
            **kwargs: Parameters derived from the class `__init__`.


        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Method process_record is not implemented in class BaseDataset")

    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        Segment all signal types in signals according to label['segments'], and return signals and labels for each segment.
        The start/end units in label['segments'] are in seconds.
        """
        segments = []
        false = False
        for seg in labels['segments']:
            seg_dict = {'signals': {}, 'labels': seg['value'], 'info': {}}
            start_time = seg['start'] + self.start_offset
            end_time = seg['end'] + self.end_offset
            seg_dict['info']= {
                'start_time': start_time,
                'end_time': end_time,
            }
            for sig_type, sig in signals.items():
                freq = sig['freq']
                data = sig['data']
                start_idx = int(round(start_time * freq))
                end_idx = int(round(end_time * freq))
                if self.include_end:
                    end_idx += 1
                if start_idx < 0 or end_idx > data.shape[-1] or start_idx > end_idx:
                    false = True
                    print(f"Invalid segment: {sig_type}, {start_time}, {end_time}, {start_idx}, {end_idx}")
                    continue
                # print(start_idx, end_idx)
                seg_dict['signals'][sig_type] = {
                    'data': data[..., start_idx:end_idx],
                    'channels': sig.get('channels', []),
                    'freq': freq,
                }
            if false:
                false = False
                continue
            segments.append(seg_dict)
        return segments

    def apply_transform(self, transforms: List, signals: dict) -> dict:
        """
        Apply a list of transforms to the signals.

        Args:
            transforms (List): A list of transform functions to apply.
            signals (dict): The signals to transform.

        Returns:
            dict: The transformed signals.
        """
        if transforms is not None:
            for transform in transforms:
                try:
                    signals = transform(signals)
                    # print(transform.__class__.__name__)
                    # print(signals['eeg']['data'])
                except Exception as e:
                    print(f"[Transform Error] {transform.__class__.__name__}: {e}")
                    return None
        return signals

    def assemble_segment(
        self, 
        key: str, 
        signals: dict, 
        labels: dict, 
        info: dict
    ) -> dict:
        """
        Build the result dictionary for a segment.
        This function is used in the 'process_record' method to assemble the
        final result for each segment.

        Args:
            key (str): The segment identifier.
            signals (dict): The signals for the segment.
            labels (dict): The labels for the segment.
            info(dict): The metadata information for the segment.

        Returns:
            dict: The result dictionary containing signals, labels, and metadata.
        """

        for sig_type, sig in signals.items():
            if 'info' not in sig or sig['info'] is None:
                sig['info'] = {}
            if 'windows' not in sig['info'] or not sig['info']['windows']:
                data_len = sig['data'].shape[-1]
                sig['info']['windows'] = [{'start': 0, 'end': data_len}]
            sig['info']['sample_ids'] = [f'{i}_{key}' for i in range(len(sig['info']['windows']))]
        
        window_lens = [len(sig['info']['windows']) for sig in signals.values()]
        if not all(l == window_lens[0] for l in window_lens):
            raise ValueError(f"{info['segment_id']}All signals must have the same number of windows, got: {window_lens}")
        sample_ids = [f'{i}_{key}' for i in range(window_lens[0])]
        
        for label_type, label in labels.items():
            if 'info' not in label or label['info'] is None:
                label['info'] = {}
            label['info']['sample_ids'] = sample_ids
        
        info['sample_ids'] = sample_ids

        result = {
            'signals': signals,
            'labels': labels,
            'key': key,
            'info': info
        }
        return result
    
    def assemble_sample(self, signals: dict, labels: dict) -> dict:
        """
        Build the result dictionary for a sample.
        This function is used in the '__getitem__' method to assemble the
        final result for each sample.

        Args:
            signals (dict): The signals for the sample.
            labels (dict): The labels for the sample.

        Returns:
            dict: The result dictionary containing signals and labels.
        """
        result = {}
        for sig_type, sig in signals.items():
            result[sig_type] = sig['data']
        for label_type, label in labels.items():
            result[label_type] = label['data']
        return result

    @property
    def repr_body(self) -> Dict:
        """
        Return the core attributes of the dataset for representation.

        Returns:
            Dict: A dictionary containing the core attributes of the dataset.
        """
        return {
            'io_path': self.io_path,
            'io_size': self.io_size,
            'io_mode': self.io_mode
        }

    @property
    def repr_tail(self) -> Dict:
        """
        Return additional attributes of the dataset for representation.

        Returns:
            Dict: A dictionary containing additional attributes of the dataset.
        """
        return {'length': self.__len__()}

    def __copy__(self) -> 'BaseDataset':
        """
        Create a shallow copy of the dataset.

        Returns:
            BaseDataset: A shallow copy of the dataset.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        # Copy basic attributes
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items()
            if k not in [
                'signal_io_router', 'label_io_router',
                'info', 'signal_paths', 'label_paths',
                'signal_cache', 'label_cache',
                'signals_info', 'labels_info'
            ]
        })

        if self.is_lazy():
            # Copy paths for lazy loading
            result.signal_paths = self.signal_paths.copy()
            result.label_paths = self.label_paths.copy()
            result.signal_io_router = None
            result.label_io_router = None
        else:
            # Eager loading copy
            result.signal_io_router = {}
            for record, signal_io in self.signal_io_router.items():
                result.signal_io_router[record] = copy(signal_io)
            result.label_io_router = {}
            for record, label_io in self.label_io_router.items():
                result.label_io_router[record] = copy(label_io)

        # Deep copy info
        result.info = copy(self.info)
        result.signals_info = copy(self.signals_info)
        result.labels_info = copy(self.labels_info)
        return result


    def __repr__(self) -> str:
        """
        Generate a string representation of the dataset.

        Returns:
            str: A string representation of the dataset.
        """
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ','
            format_string += '\n'
            # str param
            if isinstance(v, str):
                format_string += f"    {k}='{v}'"
            else:
                format_string += f"    {k}={v}"
        format_string += '\n)'
        # other info
        format_string += '\n'
        for i, (k, v) in enumerate(self.repr_tail.items()):
            if i:
                format_string += ', '
            format_string += f'{k}={v}'
        return format_string

