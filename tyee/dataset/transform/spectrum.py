#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : spectrum.py
@Time    : 2025/05/07 15:51:40
@Desc    : 
"""

import numpy as np
import mne
import pywt
from typing import Optional, Dict, Any
from scipy.fft import rfft, rfftfreq
from tyee.dataset.transform import BaseTransform

class CWTSpectrum(BaseTransform):
    """
    Continuous Wavelet Transform (CWT) time-frequency spectrum transform
    """
    def __init__(self, freqs, output_type='power', n_jobs=1, verbose=0, source=None, target=None):
        super().__init__(source, target)
        self.freqs = freqs
        self.output_type = output_type
        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, result):
        # result: dict，包含 'data'
        data = result['data']
        freq = result['freq']
        num_of_channels = data.shape[0]
        spec = mne.time_frequency.tfr_array_morlet(
            data.reshape(1, num_of_channels, -1),
            sfreq=freq,
            freqs=self.freqs,
            output=self.output_type,
            verbose=self.verbose,
            n_jobs=self.n_jobs
        )[0]  # shape: (channels, wavelets, time)
        result = result.copy()
        result['data'] = spec
        return result

class DWTSpectrum(BaseTransform):
    """
    Discrete Wavelet Transform (DWT) time-frequency spectrum transform
    """
    def __init__(self, wavelet='db4', level=4, source=None, target=None):
        super().__init__(source, target)
        self.wavelet = wavelet
        self.level = level

    def transform(self, result):
        data = result['data']
        coeffs_list = []
        for ch in range(data.shape[0]):
            coeffs = pywt.wavedec(data[ch], self.wavelet, level=self.level)
            coeffs_concat = np.concatenate([c if c.ndim > 0 else np.array([c]) for c in coeffs])
            coeffs_list.append(coeffs_concat)
        result = result.copy()
        result['data'] = np.stack(coeffs_list, axis=0)
        return result

class FFTSpectrum(BaseTransform):
    """
    Fast Fourier Transform spectrum transform with optional frequency filtering
    """
    def __init__(
        self,
        resolution: Optional[int] = None,
        min_hz: Optional[float] = None,
        max_hz: Optional[float] = None,
        axis: int = 0,
        sample_rate_key: str = 'freq',
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
        super().__init__(source, target)
        self.resolution = resolution
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.axis = axis
        self.sample_rate_key = sample_rate_key

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        sig = result['data']
        freq = result[self.sample_rate_key]
        axis = self.axis

        if self.resolution is not None:
            if sig.shape[axis] < self.resolution:
                pad_width = [(0, 0)] * sig.ndim
                pad_width[axis] = (0, self.resolution - sig.shape[axis])
                sig = np.pad(sig, pad_width, mode='constant')
            elif sig.shape[axis] > self.resolution:
                slicer = [slice(None)] * sig.ndim
                slicer[axis] = slice(0, self.resolution)
                sig = sig[tuple(slicer)]

        # FFT
        y = np.abs(rfft(sig, axis=axis))
        freqs = rfftfreq(sig.shape[axis], 1 / freq)

        if self.min_hz is not None and self.max_hz is not None:
            mask = (freqs > self.min_hz) & (freqs < self.max_hz)
            slicer = [slice(None)] * y.ndim
            slicer[axis] = mask
            y = y[tuple(slicer)]
            freqs = freqs[mask]

        result = result.copy()
        result['data'] = y
        result['freqs'] = freqs
        return result