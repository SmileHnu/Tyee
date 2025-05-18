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
from scipy.interpolate import interp1d
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
        else:
            raise ValueError("只支持2D以下信号重采样，当前shape: {}".format(signal.shape))

        # 更新采样率和信号
        result['freq'] = self.desired_freq
        result['data'] = signal_resampled

        return result

class Downsample(BaseTransform):
    """
    按采样率下采样，将信号从 cur_freq 下采样到 desired_freq，并同步更新 freq。
    """
    def __init__(self, desired_freq: int, axis: int = -1, source=None, target=None):
        super().__init__(source, target)
        self.desired_freq = desired_freq
        self.axis = axis

    def transform(self, result):
        data = result['data']
        cur_freq = result.get('freq', None)
        if cur_freq is None or cur_freq == self.desired_freq:
            return result
        length = data.shape[self.axis]
        # num_points = int(np.round(length * self.desired_freq / cur_freq))
        # indices = np.linspace(0, length - 1, num_points, dtype=int)
        indices = np.arange(0, length, cur_freq // self.desired_freq)
        # print(f"Downsampled indices: {indices}")
        data_downsampled = np.take(data, indices, axis=self.axis)
        result = result.copy()
        result['data'] = data_downsampled
        result['freq'] = self.desired_freq
        return result

class Interpolate(BaseTransform):
    """
    按采样率插值上采样，将信号从 cur_freq 插值到 desired_freq，并同步更新 freq。
    支持 'linear', 'nearest', 'cubic' 等插值方式。
    """
    def __init__(self, desired_freq: int, axis: int = -1, kind: str = 'linear', source=None, target=None):
        super().__init__(source, target)
        self.desired_freq = desired_freq
        self.axis = axis
        self.kind = kind

    def transform(self, result):
        data = np.c_[result['data'], result['data'].T[-1]]
        cur_freq = result.get('freq', None)
        if cur_freq is None or cur_freq == self.desired_freq:
            return result
        length = data.shape[self.axis]
        ratio = self.desired_freq // cur_freq
        old_indices = np.asarray(range(length)) * ratio
        new_indices = np.asarray(range(length * ratio)[
                              :-ratio])
        # print(f"Interpolated indices: {new_indices}")
        def interp_func(x):
            f = interp1d(old_indices, x, kind=self.kind, axis=0, fill_value="extrapolate")
            return f(new_indices)
        data_interp = np.apply_along_axis(interp_func, self.axis, data)
        result = result.copy()
        result['data'] = data_interp
        result['freq'] = self.desired_freq
        return result

    # def transform(self, result):
    #     data = result['data']
    #     cur_freq = result.get('freq', None)
    #     if cur_freq is None or cur_freq == self.desired_freq:
    #         return result
    #     length = data.shape[self.axis]
    #     num_points = int(np.round(length * self.desired_freq / cur_freq))
    #     old_indices = np.arange(length)
    #     new_indices = np.linspace(0, length - 1, num_points)
    #     print(f"Interpolated indices: {new_indices}")
    #     def interp_func(x):
    #         f = interp1d(old_indices, x, kind=self.kind, axis=0, fill_value="extrapolate")
    #         return f(new_indices)
    #     data_interp = np.apply_along_axis(interp_func, self.axis, data)
    #     result = result.copy()
    #     result['data'] = data_interp
    #     result['freq'] = self.desired_freq
    #     return result