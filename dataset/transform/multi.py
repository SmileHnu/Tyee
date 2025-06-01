#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : multi.py
@Time    : 2025/05/16 15:17:34
@Desc    : 
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataset.transform.base_transform import BaseTransform

class Concat(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
        super().__init__(source=source, target=target)
        self.axis = axis

    def transform(self, signals: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Any:
        # 如果 source 是列表，signals 是对应的字典列表
        if isinstance(self.source, list):
            datas = [s['data'] for s in signals]
            data = np.concatenate(datas, axis=self.axis)
            channel_names = []
            for s in signals:
                if 'channels' in s:
                    if isinstance(s['channels'], list):
                        channel_names.extend(s['channels'])
                    else:
                        channel_names.append(s['channels'])
            freq = signals[0].get('freq', None) if signals else None

            result = {'data': data}
            if freq is not None:
                result['freq'] = freq
            if channel_names:
                result['channels'] = channel_names
            return result
        else:
            # 只返回单个数据
            return signals

class Stack(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
        super().__init__(source=source, target=target)
        self.axis = axis

    def transform(self, signals: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Any:
        if isinstance(self.source, list):
            datas = [s['data'] for s in signals]
            data =  np.stack(datas, axis=self.axis)
            return {'data': data}
        else:
            return signals