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
import mne
import numpy as np
from dataset.transform import BaseTransform
from mne.filter import resample
from typing import Optional

mne.set_log_level('WARNING')

class Resample(BaseTransform):
    def __init__(self,
                 desired_freq: Optional[int] = None,
                 axis: int = -1,
                 window: str = "auto",
                 n_jobs: Optional[int] = None,
                 pad: str = "auto",
                 npad: str = 'auto',
                 method: str = "fft",
                 verbose: Optional[bool] = None,
                 source: Optional[str] = None,
                 target: Optional[str] = None):
        super().__init__(source, target)
        self.desired_freq = desired_freq
        self.axis = axis
        self.method = method
        self.window = window
        self.npad = npad
        self.n_jobs = n_jobs
        self.pad = pad
        self.verbose = verbose

    def transform(self, result):
        # 兼容新数据结构，字段为 data 和 freq
        signal = result['data']
        freq = result['freq']
        if self.desired_freq is None or freq == self.desired_freq:
            # 不需要重采样
            return result

        if signal.ndim <= 2:
            # (通道数, 时间点)
            signal_resampled = resample(
                signal,
                up=self.desired_freq,
                down=freq,
                axis=self.axis,
                window=self.window,
                n_jobs=self.n_jobs,
                pad=self.pad,
                npad=self.npad,
                method=self.method,
                verbose=self.verbose
            )
        elif signal.ndim == 3:
            # (通道数, 频率数, 时间点)
            channels, freqs, time_steps = signal.shape
            signal_reshaped = signal.reshape(channels * freqs, time_steps)
            signal_resampled = resample(
                signal_reshaped,
                up=self.desired_freq,
                down=freq,
                axis=1,
                window=self.window,
                n_jobs=self.n_jobs,
                pad=self.pad,
                npad=self.npad,
                method=self.method,
                verbose=self.verbose
            )
            new_time_steps = signal_resampled.shape[1]
            signal_resampled = signal_resampled.reshape(channels, freqs, new_time_steps)
        else:
            raise ValueError("只支持3D以下信号重采样，当前shape: {}".format(signal.shape))

        # 更新采样率和信号
        result['freq'] = self.desired_freq
        result['data'] = signal_resampled

        return result