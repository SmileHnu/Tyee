#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : meta_io.py
@Time    : 2025/02/20 13:53:23
@Desc    : 
"""

import os
import pandas as pd
import csv
from typing import Dict

class MetaInfoIO:
    r'''
    使用该类与 PhysioSignalIO 一起存储生理信号（如 EEG, ECG 等）的描述信息
    ，以表格的形式保存，用户可以在生成后继续分析、插入、删除和修改对应的信息。

    示例用法：
        meta_info_io = MetaInfoIO('YOUR_PATH')
        key = meta_info_io.write_info({
            'signal_id': 0,
            'signal_type': 'EEG',
            'sampling_rate': 1000,
            'duration': 300.0,
            'channels': ['Fp1', 'Fp2', 'C3', 'C4']  
        })
        info = meta_info_io.read_info(key).to_dict()
        >>> {
                'signal_id': 0,
                'signal_type': 'EEG',
                'sampling_rate': 1000,
                'duration': 300.0,
                'channels': ['Fp1', 'Fp2', 'C3', 'C4']  
            }

    参数：
        io_path (str): 存储元数据表格的路径。
    '''
    def __init__(self, io_path: str) -> None:
        '''
        初始化 MetaInfoIO 类。

        参数：
            io_path (str): 存储元数据表格的路径。
        '''
        self.io_path = io_path
        # 如果路径不存在，则创建目录并初始化表格
        if not os.path.exists(self.io_path):
            os.makedirs(os.path.dirname(io_path), exist_ok=True)
            open(self.io_path, 'x').close()
            self.write_pointer = 0
        else:
            self.write_pointer = len(self)

    def __len__(self):
        '''返回元数据表格中的记录数量'''
        if os.path.getsize(self.io_path) == 0:
            return 0
        info_list = pd.read_csv(self.io_path)
        return len(info_list)

    def write_info(self, obj: Dict) -> int:
        r'''
        向表格中插入生理信号的描述信息。

        参数：
            obj (dict): 要写入表格的描述信息，包含信号的各种元数据字段，例：
                'signal_id', 'signal_type', 'sampling_rate', 'duration', 'channels' 等。

        返回：
            int: 写入的描述信息在表格中的索引。
        '''
        
        if 'channels' in obj and isinstance(obj['channels'], list):
            obj['channels'] = ','.join(obj['channels'])

        with open(self.io_path, 'a+') as f:
            require_head = os.path.getsize(self.io_path) == 0  
            writer = csv.DictWriter(f, fieldnames=list(obj.keys()))  
            if require_head:
                writer.writeheader()  
            writer.writerow(obj)  
        key = self.write_pointer
        self.write_pointer += 1  
        return key

    def read_info(self, key: int) -> pd.DataFrame:
        r'''
        根据索引查询对应的生理信号描述信息。

        参数：
            key (int): 要查询的描述信息的索引。

        返回：
            pd.DataFrame: 对应的生理信号描述信息。
        '''
        info = pd.read_csv(self.io_path).iloc[key]
        # 如果 'channels' 字段是字符串，则将其转换为列表
        if 'channels' in info:
            info['channels'] = info['channels'].split(',')
        return info

    def read_all(self) -> pd.DataFrame:
        r'''
        获取表格中所有的生理信号描述信息。

        返回：
            pd.DataFrame: 所有生理信号的描述信息。
        '''
        if os.path.getsize(self.io_path) == 0:
            return pd.DataFrame()  
        df = pd.read_csv(self.io_path)
        # 如果 'channels' 字段是字符串，则将其转换为列表
        if 'channels' in df.columns:
            df['channels'] = df['channels'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        return df
