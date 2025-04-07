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
from copy import copy
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
        io_path: Union[str, None] = None,
        io_mode: str = 'lmdb',
        io_size: int = 1048576,
        lazy_threshold: int = 128,
        num_worker: int = 0,
        verbose: bool = True,
        after_trial: Callable = None,
        after_session: Callable = None,
        after_subject: Callable = None,
        signal_types: list = ['EEG'], 
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
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = io_mode
        self.lazy_threshold = lazy_threshold
        self.num_worker = num_worker
        self.verbose = verbose
        self.after_trial = after_trial
        self.after_session = after_session
        self.after_subject = after_subject
        self.signal_types = [signal_type.lower() for signal_type in signal_types]
        
        # Check if the dataset folder is empty or in memory mode
        if self.is_folder_empty(self.io_path) or self.io_mode == 'memory':
            log.info(
                f'🔍 | No cached processing results found, processing {self.signal_types} data from {self.io_path}.')
            os.makedirs(self.io_path, exist_ok=True)

            records = self.set_records(**kwargs)
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
                            self.handle_record(io_path=self.io_path,
                                               io_size=self.io_size,
                                               io_mode=self.io_mode,
                                               record=record,
                                               record_id=record_id,
                                               read_record=self.read_record,
                                               process_record=self.process_record,
                                               signal_types=self.signal_types,  
                                               verbose=self.verbose,
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
                            io_path=io_path,
                            io_size=io_size,
                            io_mode=io_mode,
                            record_id=record_id,
                            record=record,
                            read_record=self.read_record,
                            process_record=self.process_record,
                            signal_types=self.signal_types,  
                            verbose=self.verbose,
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
                    f'✅ | All processed {self.signal_types} data has been cached to {io_path}.'
                )
                log.info(
                    f'😊 | Please set \033[92mio_path\033[0m to \033[92m{io_path}\033[0m for the next run, to directly read from the cache if you wish to skip the data processing step.'
                )

            self.init_io(
                io_path=self.io_path,
                io_size=self.io_size,
                io_mode=self.io_mode)  

            if self.after_trial is not None or self.after_session is not None or self.after_subject is not None:
                # catch the exception
                print("update_record")
                try:
                    self.update_record(after_trial=after_trial,
                                       after_session=after_session,
                                       after_subject=after_subject)
                except Exception as e:
                    # shutil to delete the database
                    shutil.rmtree(self.io_path)
                    raise e
        else:
            log.info(
                f'🔍 | Detected cached processing results, reading cache from {self.io_path}.'
            )
            self.init_io(
                io_path=self.io_path,
                io_size=self.io_size,
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

    def init_io(self, io_path: str, io_size: int, io_mode: str):
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

        if len(records) > self.lazy_threshold:
            # Store paths instead of PhysioSignalIO instances
            self.signal_paths = {}
            for record in records:
                meta_info_io_path = os.path.join(io_path, record, 'info.csv')
                self.signal_paths[record] = {}
            
                signal_path = os.path.join(io_path, record)
                self.signal_paths[record] = signal_path

                info_io = MetaInfoIO(meta_info_io_path)
                info_df = info_io.read_all()
                # 检查 DataFrame 是否为空
                if info_df.empty:
                    log.warning(f"Empty DataFrame for record: {record}, skipping.")
                    continue
                
                assert 'record_id' not in info_df.columns, \
                    "column 'record_id' is a forbidden reserved word."
                info_df['record_id'] = record
                info_merged.append(info_df)

            self.signal_io_router = {}  # Not used in lazy mode
        else:
            self.signal_io_router = {}
            for record in records:
                meta_info_io_path = os.path.join(io_path, record, 'info.csv')
                self.signal_io_router[record] = {}
                
                signal_io_path = os.path.join(io_path, record)
                signal_io = PhysioSignalIO(signal_io_path,
                                        io_size=io_size,
                                        io_mode=io_mode)
                self.signal_io_router[record] = signal_io

                info_io = MetaInfoIO(meta_info_io_path)
                info_df = info_io.read_all()
                if info_df.empty:
                    log.warning(f"Empty DataFrame for record: {record}, skipping.")
                    continue
                
                assert 'record_id' not in info_df.columns, \
                    "column 'record_id' is a forbidden reserved word."
                info_df['record_id'] = record
                info_merged.append(info_df)

            self.signal_paths = {}  # Not used in eager mode

        self.info = pd.concat(info_merged, ignore_index=True)

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
        if hasattr(self, 'signal_io_router') and len(self.signal_io_router) > 0:
            return False
        if hasattr(self, 'signal_paths') and len(self.signal_paths) > 0:
            return True
        raise ValueError("Both signal_io_router and signal_paths are empty.")

    def read_signal(self, record: str, key: str, signal_type: str) -> Any:
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
                io_mode=self.io_mode
            )
            return signal_io.read_signal(signal_type, key)
        else:
            signal_io = self.signal_io_router[record]
            return signal_io.read_signal(signal_type, key)

    def write_signal(self, record: str, key: str, signal: Any, signal_type: str):
        """
        Write a signal to the dataset.

        Args:
            record (str): The record identifier.
            key (str): The key of the signal to write.
            signal (Any): The signal data to write.
            signal_type (str): The type of signal to write.
        """
        if self.is_lazy():
            # Create temporary PhysioSignalIO instance
            signal_io = PhysioSignalIO(
                self.signal_paths[record],
                io_size=self.io_size,
                io_mode=self.io_mode
            )
            signal_io.write_signal(signal=signal, signal_type=signal_type, key=key)
        else:
            signal_io = self.signal_io_router[record]
            signal_io.write_signal(signal=signal, signal_type=signal_type, key=key)

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
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])
        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        result['label'] = info['label']
        return result

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
            if isinstance(data[0], dict):
                # Process nested dictionaries
                return {key: recursive_collate([d[key] for d in data]) for key in data[0]}
            elif isinstance(data[0], torch.Tensor):
                # Stack tensors
                return torch.stack(data)
            elif isinstance(data[0], np.ndarray):
                # Convert numpy arrays to tensors
                return torch.tensor(np.stack(data))
            elif isinstance(data[0], (int, float, list)):
                # Convert scalars or lists to tensors
                return torch.tensor(data)
            else:
                # Return data as-is for unsupported types
                return data

        # Apply recursive collation to each key
        for key in batch_signals:
            batch_signals[key] = recursive_collate(batch_signals[key])

        return batch_signals  

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
            if k not in ['signal_io_router', 'info', 'signal_paths']
        })

        if self.is_lazy():
            # Copy paths for lazy loading
            result.signal_paths = self.signal_paths.copy()
            result.signal_io_router = None
        else:
            # Original eager loading copy
            result.signal_io_router = {}
            for record, signal_ios in self.signal_io_router.items():
                result.signal_io_router[record] = {}
                for signal_type, signal_io in signal_ios.items():
                    result.signal_io_router[record][signal_type] = copy(signal_io)

        # Deep copy info
        result.info = copy(self.info)
        return result

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

    @staticmethod
    def handle_record(
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_mode: str = 'lmdb',
        record: Any = None,
        record_id: Union[int, str] = None,
        read_record: Callable = None,
        process_record: Callable = None,
        signal_types: list = ['eeg'],  
        verbose: bool = True,
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
        _record_id = str(record_id)
        meta_info_io_path = os.path.join(io_path, f'record_{_record_id}', 'info.csv')
        info_io = MetaInfoIO(meta_info_io_path)

        signal_io_path = os.path.join(io_path, f'record_{_record_id}')
        signal_io = PhysioSignalIO(signal_io_path,
                                    io_size=io_size,
                                    io_mode=io_mode)

        kwargs['record'] = record
        kwargs['signal_types'] = signal_types
        kwargs['result'] = read_record(record, **kwargs)
        gen = process_record(**kwargs)

        if record_id == 0:
            pbar = tqdm(disable=not verbose,
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

            for signal_type in signal_types:
                if obj and signal_type.lower() in obj and 'key' in obj:
                    signal_io.write_signal(obj[signal_type.lower()], signal_type, obj['key'])
            if obj and 'info' in obj:
                info_io.write_info(obj['info'])

        if record_id == 0:
            pbar.close()

        return {
            'record': f'record_{_record_id}'
        }
    
    @staticmethod
    def read_record(record: str | tuple, **kwargs) -> Dict:
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
                        'taskA':{
                            'type': 'event',
                            'data': [
                                {'start': 0.5, 'end': 2.3, 'value': 'stimulus'},
                                {'start': 3.0, 'end': 4.0, 'value': 'response'}
                            ],
                        },
                        'taskB':{
                            'type': 'point',
                            'data': [
                                {'index': 1024, 'value': 'N'},  
                                {'index': 2048, 'value': 'V'},
                                {'index': 3072, 'value': 'A'}
                            ]
                        },
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

    @staticmethod
    def process_record(result: Dict, **kwargs) -> Dict:
        """
        Process a record to generate the database.

        This method describes how to process a file to generate the database. It is called
        in the `__init__` method of the class and executed in parallel using `joblib.Parallel`.

        Args:
            result (Any): The result to process. It is an Dict returned by `read_records`.
            **kwargs: Parameters derived from the class `__init__`.

        Example:
            def process_record(record: Any = None, chunk_size: int = 128, **kwargs):
                # Process the record
                eeg_signal = np.ndarray(62,chunk_size),dtype=np.float32)
                eeg_channel = ['1','2',....]
                eeg_freq = 200
                eeg = {
                    'signal': signal,
                    'channels': eeg_channel,
                    'freq': eeg_freq
                }
                key = '1'
                info = {
                    'subject': '1',
                    'session': '1',
                    'run': '1',
                    'label': '1'
                }
                yield {
                    'eeg': eeg,
                    'key': key,
                    'info': info
                }

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Method process_record is not implemented in class BaseDataset")

    def update_record(
        self,
        after_trial: Callable = None,
        after_session: Callable = None,
        after_subject: Callable = None
    ) -> None:
        """
        Apply post-processing hooks to the dataset.

        This method applies the provided hooks (`after_trial`, `after_session`, `after_subject`)
        to process the dataset at different levels (trial, session, subject). The processed
        data is written back to the dataset.

        Args:
            after_trial (Callable, optional): A hook function to process data at the trial level.
            after_session (Callable, optional): A hook function to process data at the session level.
            after_subject (Callable, optional): A hook function to process data at the subject level.
        """
        pbar = tqdm(total=len(self),
                    disable=not self.verbose,
                    desc="[POST-PROCESS]")

        if after_trial is None and after_session is None and after_subject is None:
            return

        if 'subject_id' in self.info.columns:
            subject_df = self.info.groupby('subject_id')
        else:
            subject_df = [(None, self.info)]
        if after_subject is None:
            after_subject = {signal_type: lambda x: x for signal_type in self.signal_types}

        for _, subject_info in subject_df:
            subject_record_list = []
            subject_index_list = []
            subject_samples = {signal_type: [] for signal_type in self.signal_types}

            if 'session_id' in subject_info.columns:
                session_df = subject_info.groupby('session_id')
            else:
                session_df = [(None, subject_info)]
            if after_session is None:
                after_session = {signal_type: lambda x: x for signal_type in self.signal_types}

            for _, session_info in session_df:
                if 'trial_id' in session_info.columns:
                    trial_df = session_info.groupby('trial_id')
                else:
                    trial_df = [(None, session_info)]
                    if not after_trial is None:
                        log.info(
                            "No trial_id column found in info, after_trial hook is ignored."
                        )
                if after_trial is None:
                    after_trial = {signal_type: lambda x: x for signal_type in self.signal_types}

                session_samples = {signal_type: [] for signal_type in self.signal_types}
                for _, trial_info in trial_df:
                    trial_samples = {signal_type: [] for signal_type in self.signal_types}
                    for i in range(len(trial_info)):
                        signal_index = str(trial_info.iloc[i]['clip_id'])
                        signal_record = str(trial_info.iloc[i]['record_id'])

                        subject_record_list.append(signal_record)
                        subject_index_list.append(signal_index)

                        for signal_type in self.signal_types:
                            signal = self.read_signal(signal_record, signal_index, signal_type)
                            trial_samples[signal_type] += [signal]

                        pbar.update(1)

                    for signal_type in self.signal_types:
                        trial_samples[signal_type] = self.hook_data_interface(
                            after_trial[signal_type], trial_samples[signal_type])
                        session_samples[signal_type] += trial_samples[signal_type]

                for signal_type in self.signal_types:
                    session_samples[signal_type] = self.hook_data_interface(
                        after_session[signal_type], session_samples[signal_type])
                    subject_samples[signal_type] += session_samples[signal_type]

            for signal_type in self.signal_types:
                subject_samples[signal_type] = self.hook_data_interface(after_subject[signal_type],
                                                                        subject_samples[signal_type])

            for i in range(len(subject_samples[self.signal_types[0]])):
                signal_index = str(subject_index_list[i])
                signal_record = str(subject_record_list[i])

                for signal_type in self.signal_types:
                    self.write_signal(signal_record, signal_index, subject_samples[signal_type][i], signal_type)

        pbar.close()

    @staticmethod
    def hook_data_interface(hook: Callable, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply post-processing hooks to the dataset.

        This method applies the provided hooks (`after_trial`, `after_session`, `after_subject`)
        to process the dataset at different levels (trial, session, subject). The processed
        data is written back to the dataset.

        Args:
            after_trial (Callable, optional): A hook function to process data at the trial level.
            after_session (Callable, optional): A hook function to process data at the session level.
            after_subject (Callable, optional): A hook function to process data at the subject level.
        """
        # Initialize a dictionary to store stacked data
        stacked_data = {key: [] for key in data[0].keys()}

        # Stack data into tensors
        for item in data:
            for key, value in item.items():
                stacked_data[key].append(value)

        # Convert lists to tensors or arrays
        for key, value in stacked_data.items():
            if isinstance(value[0], np.ndarray):
                stacked_data[key] = np.stack(value, axis=0)
            elif isinstance(value[0], torch.Tensor):
                stacked_data[key] = torch.stack(value, axis=0)

        
        processed_data = hook(stacked_data)

        # Split processed tensors back into individual items
        result = []
        for i in range(len(data)):
            item = {}
            for key, value in processed_data.items():
                if isinstance(value, np.ndarray):
                    value = np.split(value, value.shape[0], axis=0)
                    item[key] = [np.squeeze(v, axis=0) for v in value]
                elif isinstance(value, torch.Tensor):
                    value = np.split(value, value.shape[0], dim=0)
                    item[key] = [np.squeeze(v, axis=0) for v in value]
                else:
                    item[key] = value[i]
            result.append(item)

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

        