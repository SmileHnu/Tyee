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
    def __init__(self, io_path: str) -> None:
        self.io_path = io_path
        if not os.path.exists(self.io_path):
            os.makedirs(os.path.dirname(io_path), exist_ok=True)
            open(self.io_path, 'x').close()
            self.write_pointer = 0
        else:
            self.write_pointer = len(self)

    def __len__(self):
        if os.path.getsize(self.io_path) == 0:
            return 0
        info_list = pd.read_csv(self.io_path)
        return len(info_list)

    def write_info(self, obj: Dict) -> int:
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
        info = pd.read_csv(self.io_path).iloc[key]
        if 'channels' in info:
            info['channels'] = info['channels'].split(',')
        return info

    def read_all(self) -> pd.DataFrame:
        if os.path.getsize(self.io_path) == 0:
            return pd.DataFrame()  
        df = pd.read_csv(self.io_path)
        if 'channels' in df.columns:
            df['channels'] = df['channels'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        return df
