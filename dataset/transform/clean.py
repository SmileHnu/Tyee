#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : clean.py
@Time    : 2025/03/26 19:18:41
@Desc    : 
"""

import numpy as np
from utils import lazy_import_module
from .base_transform import BaseTransform

class Clean(BaseTransform):
    def __init__(self, type_clean: str ,method: str = 'neurokit', **kwargs):
        """
        初始化清洗变换类。

        参数:
        - type_clean: 清洗信号类型 ecg_clean, emg_clean等，见neurokit2的clean方法。
        - method: 清洗方法，'neurokit' ,参考neurokit2的具体clean方法。
        - kwargs: 清洗方法的参数。
        """
        super().__init__()
        self.type_clean = type_clean
        self.method = method
        self.kwargs = kwargs
        self.clean = lazy_import_module('neurokit2', type_clean)
    
    def transform(self, result):
        """
        对信号进行清洗处理。

        参数:
        - result: 包含信号数据的字典。

        返回:
        - 更新后的信号数据字典。
        """
        signals = result['signals']
        sampling_rate = result['sampling_rate']
        # 如果是二维数组，逐通道处理
        if len(signals.shape) == 2:
            cleaned_signals = []
            for channel in signals:
                cleaned_signals.append(self.clean(channel, sampling_rate=sampling_rate, **self.kwargs))
            result["signals"] = np.array(cleaned_signals)
        else:
            # 如果是一维数组，直接处理
            result["signals"] = self.clean(signals, sampling_rate=sampling_rate, **self.kwargs)
        return result
