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
        signal = result['data']
        freq = result['freq']
        if self.desired_freq is None or freq == self.desired_freq:
            return result

        if signal.ndim <= 2:
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
            raise ValueError("Only supports resampling for signals with 2D or lower dimensions, current shape: {}".format(signal.shape))

        result['freq'] = self.desired_freq
        result['data'] = signal_resampled

        return result

class Downsample(BaseTransform):
    """
    Downsample by sampling rate, downsample signal from cur_freq to desired_freq, and update freq accordingly.
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
        indices = np.arange(0, length, cur_freq // self.desired_freq)
        data_downsampled = np.take(data, indices, axis=self.axis)
        result = result.copy()
        result['data'] = data_downsampled
        result['freq'] = self.desired_freq
        return result

class Interpolate(BaseTransform):
    """
    Upsample by interpolation, interpolate signal from cur_freq to desired_freq, and update freq accordingly.
    Supports interpolation methods like 'linear', 'nearest', 'cubic', etc.
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