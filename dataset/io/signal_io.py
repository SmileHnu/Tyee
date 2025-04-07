#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : signal_io.py
@Time    : 2025/02/20 13:53:32
@Desc    : 
"""

import pickle
import os
from typing import Union

import torch
import lmdb

class _PhysioSignalIO:
    def get_total_length(self):
        total_length = 0
        for signal_type in self.signal_types():
            total_length += self.get_signal_length(signal_type)
        return total_length

    def get_signal_length(self, signal_type: str) -> int:
        raise NotImplementedError
    
    def signal_types(self):
        raise NotImplementedError
    
    def keys(self, signal_type: str):
        raise NotImplementedError

    def signals(self, signal_type: str):
        raise NotImplementedError

    def read_signal(self, signal_type: str, key: str) -> any:
        raise NotImplementedError

    def write_signal(self,
                  signal: Union[any, torch.Tensor],
                  signal_type: str,
                  key: Union[str, None] = None) -> str:
        raise NotImplementedError

class LMDBPhysioSignalIO(_PhysioSignalIO):
    """
    使用 LMDB 存储生理信号的实现类，按信号类型分开存储。
    """
    def __init__(self, io_path: str, io_size: int = 1048576):
        """
        初始化 LMDBPhysioSignalIO 实例。
        
        参数:
        io_path (str): 存储所有信号类型的根目录。
        io_size (int): LMDB 数据库的初始大小。
        """
        self.io_path = io_path
        self.io_size = io_size
        os.makedirs(io_path, exist_ok=True)

        # 为每个信号类型创建独立的 LMDB 数据库
        self._envs = {}  # 用于存储不同信号类型的 LMDB 环境

    def _get_env(self, signal_type: str):
        """
        根据信号类型返回相应的 LMDB 环境。如果不存在，则创建一个新的数据库环境。
        
        参数:
        signal_type (str): 信号的类型
        
        返回:
        lmdb.Environment: 对应信号类型的 LMDB 环境
        """
        if signal_type not in self._envs:
            # 创建每种信号类型的独立数据库路径
            signal_db_path = os.path.join(self.io_path, signal_type)
            os.makedirs(signal_db_path, exist_ok=True)
            self._envs[signal_type] = lmdb.open(signal_db_path, map_size=self.io_size)

        return self._envs[signal_type]

    def __del__(self):
        """
        关闭所有 LMDB 数据库。
        """
        for env in self._envs.values():
            env.close()

    def __len__(self):
        """
        返回所有信号类型数据库中的条目总数。
        
        返回:
        int: 所有信号数据库的条目总数。
        """
        total_entries = 0
        for signal_type, env in self._envs.items():
            with env.begin(write=False) as transaction:
                total_entries += transaction.stat()['entries']
        return total_entries

    def get_signal_length(self, signal_type: str) -> int:
        """
        返回指定信号类型数据库中的条目总数。
        
        参数:
        signal_type (str): 信号的类型
        
        返回:
        int: 指定信号类型数据库的条目总数。
        """
        env = self._get_env(signal_type)
        with env.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        """
        写入信号到对应信号类型的 LMDB 数据库。
        
        参数:
        signal (Union[any, torch.Tensor]): 要写入的信号。
        signal_type (str): 信号的类型。
        key (Union[str, None]): 信号的键。如果为 None，则自动生成键。
        
        返回:
        str: 写入信号的键。
        
        异常:
        RuntimeError: 如果信号为 None 或者无法写入信号。
        """
        env = self._get_env(signal_type)  # 获取对应信号类型的 LMDB 环境
        
        if key is None:
            key = f"{signal_type}_{self.get_signal_length(signal_type)}"

        if signal is None:
            raise RuntimeError(f'Save None to the LMDB with the key {key}!')

        try_again = False
        try:
            with env.begin(write=True) as transaction:
                transaction.put(f"{signal_type}_{key}".encode(), pickle.dumps(signal))
        except lmdb.MapFullError:
            self.io_size = self.io_size * 2
            env.set_mapsize(self.io_size)
            try_again = True

        if try_again:
            return self.write_signal(signal=signal, signal_type=signal_type, key=key)
        return key

    def read_signal(self, signal_type: str, key: str) -> any:
        """
        从 LMDB 数据库中读取信号。
        
        参数:
        signal_type (str): 信号的类型。
        key (str): 信号的键。
        
        返回:
        any: 读取的信号。
        
        异常:
        RuntimeError: 如果无法找到指定键的信号。
        """
        env = self._get_env(signal_type)  # 获取对应信号类型的 LMDB 环境
        
        with env.begin(write=False) as transaction:
            signal = transaction.get(f"{signal_type}_{key}".encode())
        
        if signal is None:
            raise RuntimeError(f'Unable to index the {signal_type} signal sample with key {key}!')

        return pickle.loads(signal)

    def signal_types(self):
        """
        返回所有信号的类型。
        
        返回:
        list: 包含所有信号类型的列表。
        """
        return list(self._envs.keys())

    def keys(self, signal_type: str):
        """
        返回指定类型的所有信号的键。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含所有信号键的列表。
        """
        env = self._get_env(signal_type)  # 获取对应信号类型的 LMDB 环境
        
        with env.begin(write=False) as transaction:
            return [key.decode() for key in transaction.cursor().iternext(keys=True, values=False)]

    def signals(self, signal_type: str):
        """
        返回指定类型的所有信号。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含所有信号的列表。
        """
        return [self.read_signal(signal_type, key) for key in self.keys(signal_type)]
    def __copy__(self):
        """
        创建 LMDBPhysioSignalIO 实例的浅拷贝。
        
        返回:
        LMDBPhysioSignalIO: 新的 LMDBPhysioSignalIO 实例。
        """
        # 创建新实例
        cls = self.__class__
        result = cls.__new__(cls)
        
        # 复制数据库路径和大小等属性
        result.io_path = self.io_path
        result.io_size = self.io_size
        
        # 创建新的 envs 字典并为每个信号类型打开对应的数据库环境
        result._envs = {}
        for signal_type in self._envs:
            signal_db_path = os.path.join(self.io_path, signal_type)
            result._envs[signal_type] = lmdb.open(signal_db_path, map_size=self.io_size)
        
        return result

class MemoryPhysioSignalIO(_PhysioSignalIO):
    """
    使用内存存储生理信号的实现类，按信号类型分开存储。
    """
    def __init__(self):
        """
        初始化 MemoryPhysioSignalIO 实例。
        """
        self._memory = {}  # 存储不同信号类型的数据

    def __len__(self):
        """
        返回内存中所有信号条目的总数。

        返回:
            int: 所有信号条目的总数。
        """
        return sum(len(signals) for signals in self._memory.values())
    
    def get_signal_length(self, signal_type: str) -> int:
        """
        返回指定信号类型的条目总数。

        参数:
        signal_type (str): 信号的类型。

        返回:
        int: 指定信号类型的条目总数。
        """
        if signal_type not in self._memory:
            return 0
        return len(self._memory[signal_type])
    
    def signal_types(self):
        """
        返回所有信号的类型。

        返回:
            list: 包含所有信号类型的列表。
        """
        return list(self._memory.keys())

    def keys(self, signal_type: str):
        """
        返回指定类型的所有信号的键。

        参数:
        signal_type (str): 信号的类型。

        返回:
        list: 包含所有信号键的列表。
        """
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        return list(self._memory[signal_type].keys())

    def signals(self, signal_type: str):
        """
        返回指定类型的所有信号。

        参数:
        signal_type (str): 信号的类型。

        返回:
        list: 包含所有信号的列表。
        """
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        return list(self._memory[signal_type].values())

    def read_signal(self, signal_type: str, key: str) -> any:
        """
        从内存中读取指定类型和键的信号。

        参数:
        signal_type (str): 信号的类型。
        key (str): 信号的键。

        返回:
        any: 读取的信号。
        """
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        
        if key not in self._memory[signal_type]:
            raise RuntimeError(f"Unable to index the {signal_type} signal sample with key {key}!")

        return self._memory[signal_type][key]

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        """
        将信号写入内存中。

        参数:
        signal (Union[any, torch.Tensor]): 要写入的信号。
        signal_type (str): 信号的类型。
        key (Union[str, None]): 信号的键，如果为 None，则自动生成键。

        返回:
        str: 写入信号的键。
        """
        if signal_type not in self._memory:
            self._memory[signal_type] = {}

        if key is None:
            key = str(len(self._memory[signal_type]))

        if signal is None:
            raise RuntimeError(f'Save None to the memory with the key {key}!')

        self._memory[signal_type][key] = signal
        return key

    def __copy__(self):
        """
        复制 MemoryPhysioSignalIO 实例。
        
        返回:
        MemoryPhysioSignalIO: 复制的实例。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result._memory = self._memory.copy()
        return result


class PicklePhysioSignalIO(_PhysioSignalIO):
    """
    使用 Pickle 存储生理信号的实现类，按信号类型将数据存储在不同的文件夹中。
    """
    def __init__(self, io_path: str) -> None:
        """
        初始化 PicklePhysioSignalIO 实例。
        
        参数:
        io_path (str): 存储信号的根目录路径。
        """
        self.io_path = io_path
        os.makedirs(self.io_path, exist_ok=True)

    def __len__(self):
        """
        返回文件夹中的信号总数。
        
        返回:
        int: 文件夹中存储的信号总数。
        """
        total_count = 0
        for signal_type in self.signal_types():
            signal_folder = os.path.join(self.io_path, signal_type)
            if os.path.exists(signal_folder):
                total_count += len(os.listdir(signal_folder))
        return total_count

    def get_signal_length(self, signal_type: str) -> int:
        """
        返回指定信号类型的条目总数。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        int: 指定信号类型的条目总数。
        """
        signal_folder = os.path.join(self.io_path, signal_type)
        if not os.path.exists(signal_folder):
            return 0
        return len(os.listdir(signal_folder))
    
    def signal_types(self):
        """
        获取文件夹中所有的信号类型（即文件夹名称）。
        
        返回:
        list: 包含所有信号类型的列表。
        """
        return [d for d in os.listdir(self.io_path) if os.path.isdir(os.path.join(self.io_path, d))]

    def keys(self, signal_type: str):
        """
        获取指定信号类型的所有键。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含指定信号类型下所有键的列表。
        """
        signal_folder = os.path.join(self.io_path, signal_type)
        if not os.path.exists(signal_folder):
            raise RuntimeError(f"Signal type {signal_type} does not exist!")

        return os.listdir(signal_folder)

    def signals(self, signal_type: str):
        """
        获取指定信号类型的所有信号。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含指定信号类型的所有信号列表。
        """
        return [self.read_signal(signal_type, key) for key in self.keys(signal_type)]

    def read_signal(self, signal_type: str, key: str) -> any:
        """
        从 Pickle 文件中读取指定信号类型和键的信号。
        
        参数:
        signal_type (str): 信号的类型。
        key (str): 信号的键。
        
        返回:
        any: 读取的信号。
        """
        signal_folder = os.path.join(self.io_path, signal_type)
        signal_path = os.path.join(signal_folder, key)

        if not os.path.exists(signal_path):
            raise RuntimeError(f"Unable to find the {signal_type} signal with key {key} in {self.io_path}!")

        with open(signal_path, 'rb') as f:
            signal = pickle.load(f)
        return signal

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        """
        将信号写入文件夹中。

        参数:
        signal (Union[any, torch.Tensor]): 要写入的信号。
        signal_type (str): 信号的类型。
        key (Union[str, None]): 信号的键。如果为 None，则自动生成键。

        返回:
        str: 写入信号的键。
        
        异常:
        RuntimeError: 如果信号为 None 或无法保存信号。
        """
        # 确保信号类型的文件夹存在
        signal_folder = os.path.join(self.io_path, signal_type)
        os.makedirs(signal_folder, exist_ok=True)

        if key is None:
            key = str(len(os.listdir(signal_folder)))

        if signal is None:
            raise RuntimeError(f'Save None to the folder with the key {key}!')

        signal_path = os.path.join(signal_folder, key)
        with open(signal_path, 'wb') as f:
            pickle.dump(signal, f)

        return key

    def __copy__(self):
        """
        复制 PicklePhysioSignalIO 实例。
        
        返回:
        PicklePhysioSignalIO: 复制的实例。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    

class PhysioSignalIO:
    """
    生理信号输入输出类，根据不同的存储模式（LMDB、内存、Pickle）进行信号的读写操作。
    """
    def __init__(self,
                 io_path: str,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb'):
        """
        初始化 PhysioSignalIO 实例。
        
        参数:
        io_path (str): 存储信号的路径。
        io_size (int): 存储大小，默认为 1048576。
        io_mode (str): 存储模式，默认为 'lmdb'。可选值为 'lmdb'、'memory'、'pickle'。
        """
        self.io_mode = io_mode
        self.io_path = io_path
        self.io_size = io_size

        if self.io_mode == 'lmdb':
            self._io = LMDBPhysioSignalIO(io_path=self.io_path, io_size=self.io_size)
        elif self.io_mode == 'memory':
            self._io = MemoryPhysioSignalIO()
        elif self.io_mode == 'pickle':
            self._io = PicklePhysioSignalIO(io_path=self.io_path)
        else:
            raise ValueError(f'Unsupported IO mode: {self.io_mode}')
    
    def __del__(self):
        """
        删除 PhysioSignalIO 实例时，删除内部的 IO 实例。
        """
        del self._io

    def __len__(self):
        """
        返回信号的总数。
        
        返回:
        int: 信号的总数。
        """
        return len(self._io)
    
    def signal_types(self):
        """
        返回所有信号的类型。
        
        返回:
        list: 所有信号类型的列表。
        """
        return self._io.signal_types()
    
    def keys(self, signal_type: str):
        """
        返回指定类型的所有信号的键。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含所有信号键的列表。
        """
        return self._io.keys(signal_type)
    
    def signals(self, signal_type: str):
        """
        返回指定类型的所有信号。
        
        参数:
        signal_type (str): 信号的类型。
        
        返回:
        list: 包含所有信号的列表。
        """
        return self._io.signals(signal_type)
    
    def read_signal(self, signal_type: str, key: str) -> any:
        """
        读取指定类型和键的信号。
        
        参数:
        signal_type (str): 信号的类型。
        key (str): 信号的键。
        
        返回:
        any: 读取的信号。
        """
        return self._io.read_signal(signal_type, key)
    
    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        """
        写入信号。
        
        参数:
        signal (Union[any, torch.Tensor]): 要写入的信号。
        signal_type (str): 信号的类型。
        key (Union[str, None]): 信号的键。如果为 None，则自动生成键。
        
        返回:
        str: 写入信号的键。
        """
        return self._io.write_signal(signal, signal_type, key)
    
    def __copy__(self):
        """
        创建 PhysioSignalIO 实例的浅拷贝。
        
        返回:
        PhysioSignalIO: 新的 PhysioSignalIO 实例。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != '_io'
        })
        result._io = self._io.__copy__()
        return result
    
    def to_lmdb(self, io_path: str, io_size: int = 1048576):
        """
        将当前存储模式转换为 LMDB 模式。
        
        参数:
        io_path (str): LMDB 存储路径。
        io_size (int): LMDB 存储大小，默认为 1048576。
        """
        _io = LMDBPhysioSignalIO(io_path=io_path, io_size=io_size)

        for signal_type in self.signal_types():
            for key in self.keys(signal_type):
                signal = self.read_signal(signal_type, key)
                _io.write_signal(signal, signal_type, key)
        
        self.io_path = io_path
        self.io_size = io_size
        self.io_mode = 'lmdb'
        self._io = _io
    
    def to_memory(self):
        """
        将当前存储模式转换为内存模式。
        """
        _io = MemoryPhysioSignalIO()

        for signal_type in self.signal_types():
            for key in self.keys(signal_type):
                signal = self.read_signal(signal_type, key)
                _io.write_signal(signal, signal_type, key)
        
        self.io_mode = 'memory'
        self._io = _io
    
    def to_pickle(self, io_path: str):
        """
        将当前存储模式转换为 Pickle 模式。
        
        参数:
        io_path (str): Pickle 存储路径。
        """
        _io = PicklePhysioSignalIO(io_path=io_path)

        for signal_type in self.signal_types():
            for key in self.keys(signal_type):
                signal = self.read_signal(signal_type, key)
                _io.write_signal(signal, signal_type, key)
        
        self.io_path = io_path
        self.io_mode = 'pickle'
        self._io = _io