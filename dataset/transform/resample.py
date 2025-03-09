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
from neurokit2.signal import signal_resample

class Resample(BaseTransform):
    def __init__(self,
                desired_length=None,
                desired_sampling_rate=None,
                method="FFT"):
        super().__init__()
        self.desired_length = desired_length
        self.desired_sampling_rate = desired_sampling_rate
        self.method = method
    
    def transform(self, signal_type, result):
        # print('执行了Resample')
        signal = result[signal_type].T
        sampling_rate = result[f'{signal_type}_sampling_rate']
        print(signal.shape)
        if self.desired_length is None:
            self.desired_length = int(np.round(len(signal) * self.desired_sampling_rate / sampling_rate)) 
        result[signal_type] = signal_resample(signal, 
                                desired_length=self.desired_length, 
                                sampling_rate=sampling_rate, 
                                desired_sampling_rate=self.desired_sampling_rate, 
                                method=self.method).T
        print(result[signal_type].shape)
        if f'{signal_type}_times' in result:
            times = result[f'{signal_type}_times']
            start_time = times[0]
            print(len(times))
            
            times = np.linspace(start_time, 
                                start_time + self.desired_length / self.desired_sampling_rate, 
                                self.desired_length, 
                                endpoint=False)
            result[f'{signal_type}_times'] = times

        result[f'{signal_type}_sampling_rate'] = self.desired_sampling_rate
        
        return result