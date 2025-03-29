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
    def __init__(self,
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
                 **kwargs):
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
        
        # 判断io_path是否存在
        if not os.path.exists(self.io_path) or self.io_mode == 'memory':
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
                            signal_types=self.signal_types,  # 传递信号类型
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
        
       

    def init_io(self, io_path: str, io_size: int, io_mode: str):
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
                # 检查 DataFrame 是否为空
                if info_df.empty:
                    log.warning(f"Empty DataFrame for record: {record}, skipping.")
                    continue
                
                assert 'record_id' not in info_df.columns, \
                    "column 'record_id' is a forbidden reserved word."
                info_df['record_id'] = record
                info_merged.append(info_df)

            self.signal_paths = {}  # Not used in eager mode

        self.info = pd.concat(info_merged, ignore_index=True)

    

    def is_lazy(self):
        assert hasattr(self, 'signal_io_router') or hasattr(
            self, 'signal_paths'), "The dataset should contain signal_io_router or signal_paths."
        if hasattr(self, 'signal_io_router') and len(self.signal_io_router) > 0:
            return False
        if hasattr(self, 'signal_paths') and len(self.signal_paths) > 0:
            return True
        raise ValueError("Both signal_io_router and signal_paths are empty.")

    def read_signal(self, record: str, key: str, signal_type: str) -> Any:
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
        '''
        根据给定的 :obj:`index` 查询 MetaInfoIO 中对应的元信息。

        在元信息中，clip_id 是必需的。指定 EEG 在 EEGSignalIO 中的对应键，可以基于 :obj:`self.read_eeg(key)` 索引 EEG 样本。

        参数:
            index (int): 要查询的元信息的索引。

        返回:
            dict: 元信息。
        '''
        info = self.info.iloc[index].to_dict()
        return info

    def exist(self, io_path: str) -> bool:
        '''
        检查数据库 IO 是否存在。

        参数:
            io_path (str): 数据库 IO 的路径。

        返回:
            bool: 如果数据库 IO 存在，则返回 True，否则返回 False。
        '''

        return os.path.exists(io_path)

    def __getitem__(self, index: int) -> any:
        info = self.read_info(index)
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])
        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        result['label'] = info['label']
        return result

    def get_labels(self) -> list:
        '''
        获取数据集的标签。

        返回:
            list: 标签列表。
        '''
        labels = []
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.info)

    def __copy__(self) -> 'BaseDataset':
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

    def set_records(self, **kwargs):
        '''
        生成数据库的块方法。用于描述需要处理哪些数据块以生成数据库。在类的 :obj:`__init__` 中由 :obj:`joblib.Parallel` 并行调用。

        参数:
            lock (joblib.parallel.Lock): IO 写入器的锁。 (默认: :obj:`None`)
            **kwargs: 从类的 __init__ 派生的参数。

        .. code-block:: python

        def set_records(self, root_path: str = None, **kwargs):
                # 例如，返回文件名列表以供 process_record 处理
                return os.listdir(root_path)

        '''
        raise NotImplementedError(
            "Method set_records is not implemented in class BaseDataset")

    @staticmethod
    def handle_record(io_path: Union[None, str] = None,
                    io_size: int = 1048576,
                    io_mode: str = 'lmdb',
                    record: Any = None,
                    record_id: Union[int, str] = None,
                    read_record: Callable = None,
                    process_record: Callable = None,
                    signal_types: list = ['eeg'],  
                    verbose: bool = True,
                    **kwargs):
        _record_id = str(record_id)
        meta_info_io_path = os.path.join(io_path, f'record_{_record_id}', 'info.csv')
        info_io = MetaInfoIO(meta_info_io_path)

        signal_io_path = os.path.join(io_path, f'record_{_record_id}')
        signal_io = PhysioSignalIO(signal_io_path,
                                    io_size=io_size,
                                    io_mode=io_mode)

        kwargs['record'] = record
        kwargs['signal_types'] = signal_types
        kwargs['result'] = read_record(**kwargs)
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
    def read_record(record: Any, **kwargs) -> Dict:
        '''
        读取数据库的 IO 方法。用于描述如何读取文件。在类的 :obj:`__init__` 中由 :obj:`joblib.Parallel` 并行调用，此方法的输出将传递给 :obj:`process_record`。

        参数:
            record (Any): 要处理的记录。它是 set_records 返回的列表中的一个元素。 (默认: :obj:`Any`)
            **kwargs: 从类的 :obj:`__init__` 派生的参数。

        .. code-block:: python

            def read_record(**kwargs):
                return {
                    'samples': ...
                    'labels': ...
                }

        '''

        raise NotImplementedError(
            "Method read_record is not implemented in class BaseDataset"
        )

    @staticmethod
    def process_record(record: Any, **kwargs) -> Dict:
        '''
        生成数据库的 IO 方法。用于描述如何处理文件以生成数据库。在类的 :obj:`__init__` 中由 :obj:`joblib.Parallel` 并行调用。

        参数:
            record (Any): 要处理的记录。它是 set_records 返回的列表中的一个元素。 (默认: :obj:`Any`)
            **kwargs: 从类的 :obj:`__init__` 派生的参数。

        .. code-block:: python

            def process_record(record: Any = None, chunk_size: int = 128, **kwargs):
                # 处理记录
                eeg = np.ndarray((chunk_size, 64, 128), dtype=np.float32)
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

        '''
        raise NotImplementedError(
            "Method process_record is not implemented in class BaseDataset")

    def update_record(self,
                    after_trial: Callable = None,
                    after_session: Callable = None,
                    after_subject: Callable = None):
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
        # 初始化一个空字典，用于存储堆叠后的数据
        stacked_data = {key: [] for key in data[0].keys()}

        # 遍历每个字典，将数据堆叠到一起
        for item in data:
            for key, value in item.items():
                stacked_data[key].append(value)

        # 将列表转换为张量
        for key, value in stacked_data.items():
            if isinstance(value[0], np.ndarray):
                stacked_data[key] = np.stack(value, axis=0)
            elif isinstance(value[0], torch.Tensor):
                stacked_data[key] = torch.stack(value, axis=0)

        
        processed_data = hook(stacked_data)

        # 将处理后的张量转换回列表形式
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
        return {
            'io_path': self.io_path,
            'io_size': self.io_size,
            'io_mode': self.io_mode
        }

    @property
    def repr_tail(self) -> Dict:
        return {'length': self.__len__()}

    def __repr__(self) -> str:
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

        