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
from typing import Union, List, Dict, Any

import torch
import lmdb
import h5py
import numpy as np
import logging
log = logging.getLogger(__name__)

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
    
    def write_signals_batch(self, signals: list, signal_type: str):
        for item in signals:
            self.write_signal(item['signal'], signal_type, item['key'])

class LMDBPhysioSignalIO(_PhysioSignalIO):
    def __init__(self, io_path: str, io_size: int = 1048576):
        self.io_path = io_path
        self.io_size = io_size
        os.makedirs(io_path, exist_ok=True)

        self._envs = {}  

    def _get_env(self, signal_type: str):
        if signal_type not in self._envs:
            # 创建每种信号类型的独立数据库路径
            signal_db_path = os.path.join(self.io_path, signal_type)
            os.makedirs(signal_db_path, exist_ok=True)
            self._envs[signal_type] = lmdb.open(signal_db_path, map_size=self.io_size, lock=False)

        return self._envs[signal_type]

    def write_signals_batch(self, signals: List[Dict], signal_type: str):
        env = self._get_env(signal_type)
        with env.begin(write=True) as transaction:
            for item in signals:
                key = item.get('key')
                signal = item.get('signal')

                if key is None:
                    # This might not be thread-safe if keys are generated based on length
                    key = f"{signal_type}_{transaction.stat()['entries'] + 1}"
                
                if signal is None:
                    log.warning(f'Attempted to save None to LMDB with key {key}. Skipping.')
                    continue
                
                transaction.put(f"{signal_type}_{key}".encode(), pickle.dumps(signal))

    def __del__(self):
        for env in self._envs.values():
            env.close()

    def __len__(self):
        total_entries = 0
        for signal_type, env in self._envs.items():
            with env.begin(write=False) as transaction:
                total_entries += transaction.stat()['entries']
        return total_entries

    def get_signal_length(self, signal_type: str) -> int:
        env = self._get_env(signal_type)
        with env.begin(write=False) as transaction:
            return transaction.stat()['entries']

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        env = self._get_env(signal_type)  
        
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

    def read_signal(self, signal_type: str, key: str, start: int, end: int) -> any:
        env = self._get_env(signal_type)  
        
        with env.begin(write=False) as transaction:
            signal = transaction.get(f"{signal_type}_{key}".encode())
        
        if signal is None:
            raise RuntimeError(f'Unable to index the {signal_type} signal sample with key {key}!')

       
        signal = pickle.loads(signal)
        axis = signal.get('axis', -1)
        if start is not None or end is not None:
            slicer = [slice(None)] * signal['data'].ndim
            slicer[axis] = slice(start, end)
            signal['data'] = signal['data'][tuple(slicer)]
        return signal

    def signal_types(self):
        return [d for d in os.listdir(self.io_path) if os.path.isdir(os.path.join(self.io_path, d))]


    def keys(self, signal_type: str):
        env = self._get_env(signal_type)  
        
        with env.begin(write=False) as transaction:
            return [key.decode() for key in transaction.cursor().iternext(keys=True, values=False)]

    def signals(self, signal_type: str):
        return [self.read_signal(signal_type, key) for key in self.keys(signal_type)]
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        
        result.io_path = self.io_path
        result.io_size = self.io_size
        
        result._envs = {}
        for signal_type in self._envs:
            signal_db_path = os.path.join(self.io_path, signal_type)
            result._envs[signal_type] = lmdb.open(signal_db_path, map_size=self.io_size)
        
        return result

class MemoryPhysioSignalIO(_PhysioSignalIO):
    def __init__(self):
        self._memory = {}  

    def __len__(self):
        return sum(len(signals) for signals in self._memory.values())
    
    def get_signal_length(self, signal_type: str) -> int:
        if signal_type not in self._memory:
            return 0
        return len(self._memory[signal_type])
    
    def signal_types(self):
        return list(self._memory.keys())

    def keys(self, signal_type: str):
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        return list(self._memory[signal_type].keys())

    def signals(self, signal_type: str):
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        return list(self._memory[signal_type].values())

    def read_signal(self, signal_type: str, key: str, start: int, end: int) -> any:
        if signal_type not in self._memory:
            raise RuntimeError(f"Signal type {signal_type} does not exist!")
        
        if key not in self._memory[signal_type]:
            raise RuntimeError(f"Unable to index the {signal_type} signal sample with key {key}!")

        signal = self._memory[signal_type][key]
        axis = signal.get('axis', -1)
        if start is not None or end is not None:
            slicer = [slice(None)] * signal['data'].ndim
            slicer[axis] = slice(start, end)
            signal['data'] = signal['data'][tuple(slicer)]
        return signal

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        if signal_type not in self._memory:
            self._memory[signal_type] = {}

        if key is None:
            key = str(len(self._memory[signal_type]))

        if signal is None:
            raise RuntimeError(f'Save None to the memory with the key {key}!')

        self._memory[signal_type][key] = signal
        return key

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result._memory = self._memory.copy()
        return result


class PicklePhysioSignalIO(_PhysioSignalIO):
    def __init__(self, io_path: str) -> None:
        self.io_path = io_path
        os.makedirs(self.io_path, exist_ok=True)

    def __len__(self):
        total_count = 0
        for signal_type in self.signal_types():
            signal_folder = os.path.join(self.io_path, signal_type)
            if os.path.exists(signal_folder):
                total_count += len(os.listdir(signal_folder))
        return total_count

    def get_signal_length(self, signal_type: str) -> int:
        signal_folder = os.path.join(self.io_path, signal_type)
        if not os.path.exists(signal_folder):
            return 0
        return len(os.listdir(signal_folder))
    
    def signal_types(self):
        return [d for d in os.listdir(self.io_path) if os.path.isdir(os.path.join(self.io_path, d))]

    def keys(self, signal_type: str):
        signal_folder = os.path.join(self.io_path, signal_type)
        if not os.path.exists(signal_folder):
            raise RuntimeError(f"Signal type {signal_type} does not exist!")

        return os.listdir(signal_folder)

    def signals(self, signal_type: str):
        return [self.read_signal(signal_type, key) for key in self.keys(signal_type)]

    def read_signal(self, signal_type: str, key: str, start: int, end: int) -> any:
        signal_folder = os.path.join(self.io_path, signal_type)
        signal_path = os.path.join(signal_folder, key)

        if not os.path.exists(signal_path):
            raise RuntimeError(f"Unable to find the {signal_type} signal with key {key} in {self.io_path}!")

        with open(signal_path, 'rb') as f:
            signal = pickle.load(f)
        
        axis = signal.get('axis', -1)
        if start is not None or end is not None:
            slicer = [slice(None)] * signal['data'].ndim
            slicer[axis] = slice(start, end)
            signal['data'] = signal['data'][tuple(slicer)]
        return signal

    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
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
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
class HDF5PhysioSignalIO(_PhysioSignalIO):
    def __init__(self, io_path: str, io_chunks: int = None):
        self.io_path = io_path
        self.io_chunks = io_chunks
        self._file_handlers = {}  

    def _get_file_path(self, signal_type: str):
        folder = os.path.join(self.io_path, signal_type)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, 'data.h5')

    def _get_file(self, signal_type: str, mode='r'):
        if (signal_type, mode) not in self._file_handlers or not self._file_handlers[(signal_type, mode)].id:
            file_path = self._get_file_path(signal_type)
            self._file_handlers[(signal_type, mode)] = h5py.File(file_path, mode)
        return self._file_handlers[(signal_type, mode)]

    def close(self):
        for f in self._file_handlers.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handlers.clear()

    def __del__(self):
        self.close()

    def write_signal(self, signal: dict, signal_type: str, key: str = None) -> str:
        f = self._get_file(signal_type, mode='a')
        group = f.require_group('signals')
        if key is None:
            key = str(len(group))
        if key in group:
            raise RuntimeError(f"Key {key} already exists in {signal_type}!")
        data = signal['data']
        axis = signal.get('axis', -1)
        if self.io_chunks is not None and isinstance(data, np.ndarray):
            chunks = list(data.shape)
            chunks[axis] = self.io_chunks if self.io_chunks < data.shape[axis] else data.shape[axis]
            chunks = tuple(chunks)
        else:
            chunks = None
        dset = group.create_dataset(key, data=data, chunks=chunks)
        dset.attrs['freq'] = signal.get('freq', -1)
        dset.attrs['channels'] = np.array(signal.get('channels', []), dtype='S')
        dset.attrs['axis'] = axis
        f.flush()
        return key

    def write_signals_batch(self, signals: list, signal_type: str):
        """
        Write a batch of signals to the HDF5 file.
        'signals' is a list of dictionaries, each with 'key' and 'signal'
        """
        f = self._get_file(signal_type, mode='a')
        group = f.require_group('signals')

        for item in signals:
            key = item['key']
            signal = item['signal']

            if key in group:
                # Or log a warning, or update, depending on desired behavior
                print(f"Warning: Key {key} already exists in {signal_type}. Skipping.")
                continue

            data = signal['data']
            axis = signal.get('axis', -1)
            
            if self.io_chunks is not None and isinstance(data, np.ndarray):
                chunks = list(data.shape)
                chunks[axis] = self.io_chunks if self.io_chunks < data.shape[axis] else data.shape[axis]
                chunks = tuple(chunks)
            else:
                chunks = None

            dset = group.create_dataset(key, data=data, chunks=chunks)
            dset.attrs['freq'] = signal.get('freq', -1)
            dset.attrs['channels'] = np.array(signal.get('channels', []), dtype='S')
            dset.attrs['axis'] = axis
        
        f.flush()

    def read_signal(self, signal_type: str, key: str, start: int = None, end: int = None) -> dict:
        f = self._get_file(signal_type, mode='r')
        group = f['signals']
        if key not in group:
            raise RuntimeError(f"Unable to index the {signal_type} signal sample with key {key}!")
        dset = group[key]
        axis = dset.attrs.get('axis', -1)
        if start is not None or end is not None:
            slicer = [slice(None)] * dset.ndim
            slicer[axis] = slice(start, end)
            data = dset[tuple(slicer)]
        else:
            data = dset[()]
        if isinstance(data, bytes):
            data = data.decode()
        freq = dset.attrs['freq']
        channels = [ch.decode() for ch in dset.attrs['channels']]
        return {'data': data, 'freq': freq, 'channels': channels}

    def get_signal_length(self, signal_type: str) -> int:
        f = self._get_file(signal_type, mode='r')
        return len(f['signals'])

    def signal_types(self):
        return [d for d in os.listdir(self.io_path) if os.path.isdir(os.path.join(self.io_path, d))]

    def keys(self, signal_type: str):
        f = self._get_file(signal_type, mode='r')
        return list(f['signals'].keys())

    def signals(self, signal_type: str):
        f = self._get_file(signal_type, mode='r')
        result = []
        for key in f['signals'].keys():
            dset = f['signals'][key]
            data = dset[()]
            freq = dset.attrs['freq']
            channels = [ch.decode() for ch in dset.attrs['channels']]
            result.append({'data': data, 'freq': freq, 'channels': channels})
        return result

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.io_path = self.io_path
        result._file_handlers = {}
        return result

class PhysioSignalIO:
    def __init__(
        self,
        io_path: str,
        io_size: int = 1048576,
        io_mode: str = 'lmdb',
        io_chunks: int = None
    ):
        self.io_mode = io_mode
        self.io_path = io_path
        self.io_size = io_size
        self.io_chunks = io_chunks

        if self.io_mode == 'lmdb':
            self._io = LMDBPhysioSignalIO(io_path=self.io_path, io_size=self.io_size)
        elif self.io_mode == 'memory':
            self._io = MemoryPhysioSignalIO()
        elif self.io_mode == 'pickle':
            self._io = PicklePhysioSignalIO(io_path=self.io_path)
        elif self.io_mode == 'hdf5':
            self._io = HDF5PhysioSignalIO(io_path=self.io_path, io_chunks=self.io_chunks)
        else:
            raise ValueError(f'Unsupported IO mode: {self.io_mode}')
    
    def __del__(self):
        del self._io

    def __len__(self):
        return len(self._io)
    
    def signal_types(self):
        return self._io.signal_types()
    
    def keys(self, signal_type: str):
        return self._io.keys(signal_type)
    
    def signals(self, signal_type: str):
        return self._io.signals(signal_type)
    
    def read_signal(self, signal_type: str, key: str, start: int, end:int) -> any:
        return self._io.read_signal(signal_type, key, start, end)
    
    def write_signal(self, signal: Union[any, torch.Tensor], signal_type: str, key: Union[str, None] = None) -> str:
        return self._io.write_signal(signal, signal_type, key)
    
    def write_signals_batch(self, signals: list, signal_type: str):
        return self._io.write_signals_batch(signals, signal_type)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update({
            k: v
            for k, v in self.__dict__.items() if k != '_io'
        })
        result._io = self._io.__copy__()
        return result