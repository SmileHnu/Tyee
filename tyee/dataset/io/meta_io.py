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
            self.info_df = pd.DataFrame()
        else:
            if os.path.getsize(self.io_path) == 0:
                self.info_df = pd.DataFrame()
                self.write_pointer = 0
            else:
                self.info_df = pd.read_csv(self.io_path)
                self.write_pointer = len(self.info_df)

    def __len__(self):
        if self.info_df is not None and not self.info_df.empty:
            return len(self.info_df)
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
        info = self.info_df.iloc[key]
        if 'channels' in info and isinstance(info['channels'], str):
            info['channels'] = info['channels'].split(',')
        return info

    def read_all(self) -> pd.DataFrame:
        df = self.info_df.copy()
        if 'channels' in df.columns:
            df['channels'] = df['channels'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
        return df
