#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : resample.py
@Time    : 2025/03/03 14:47:47
@Desc    : 
"""

import numpy as np
from dataset.transform import BaseTransform
from mne.filter import resample

class Resample(BaseTransform):
    def __init__(self,
                desired_sampling_rate=None,
                method="fft"):
        super().__init__()
        self.desired_sampling_rate = desired_sampling_rate
        self.method = method
    
    def transform(self, result):
        # 获取信号和采样率
        signal = result['signals']
        sampling_rate = result['sampling_rate']
        
        # 使用 mne 的 resample 函数进行重采样
        result['signals'] = resample(signal, up=self.desired_sampling_rate, down=sampling_rate, npad='auto', method=self.method)
        
        # 更新时间戳
        if 'times' in result:
            times = result['times']
            start_time = times[0]
            num_samples = result['signals'].shape[1]
            times = np.linspace(start_time, 
                                start_time + (num_samples) / self.desired_sampling_rate, 
                                num_samples, 
                                endpoint=True)
            result['times'] = times

        # 更新采样率
        result['sampling_rate'] = self.desired_sampling_rate
        
        return result