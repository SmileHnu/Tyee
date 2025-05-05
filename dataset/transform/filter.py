#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : filter.py
@Time    : 2025/03/03 14:14:37
@Desc    : 
"""

import mne
import numpy as np
from typing import Dict, Any, List, Optional
from dataset.transform import BaseTransform
from scipy.signal import cheby2, filtfilt
mne.set_log_level('WARNING')

class Filter(BaseTransform):
    def __init__(self, 
                 l_freq = None, 
                 h_freq = None, 
                 filter_length="auto", 
                 l_trans_bandwidth="auto", 
                 h_trans_bandwidth="auto", 
                 method="fir", 
                 iir_params=None, 
                 phase="zero", 
                 fir_window="hamming", 
                 fir_design="firwin", 
                 pad="reflect_limited",
                 source: Optional[str] = None,
                 target: Optional[str] = None):
        super().__init__(source, target)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.filter_length = filter_length
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.method = method
        self.iir_params = iir_params
        self.phase = phase
        self.fir_window = fir_window
        self.fir_design = fir_design
        self.pad = pad

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        signal_data = result['data']
        sfreq = result['freq']
        filtered_signal = mne.filter.filter_data(
            data=signal_data,
            sfreq=sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            filter_length=self.filter_length,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth,
            method=self.method,
            iir_params=self.iir_params,
            phase=self.phase,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
            pad=self.pad
        )
        result['data'] = filtered_signal
        return result

class NotchFilter(BaseTransform):
    def __init__(self, 
                 freqs: List[float], 
                 filter_length="auto", 
                 notch_widths=None, 
                 trans_bandwidth=1, 
                 method="fir", 
                 iir_params=None, 
                 mt_bandwidth=None, 
                 p_value=0.05, 
                 phase="zero", 
                 fir_window="hamming", 
                 fir_design="firwin", 
                 pad="reflect_limited",
                 source: Optional[str] = None,
                 target: Optional[str] = None):
        super().__init__(source, target)
        self.freqs = freqs
        self.filter_length = filter_length
        self.notch_widths = notch_widths
        self.trans_bandwidth = trans_bandwidth
        self.method = method
        self.iir_params = iir_params
        self.mt_bandwidth = mt_bandwidth
        self.p_value = p_value
        self.phase = phase
        self.fir_window = fir_window
        self.fir_design = fir_design
        self.pad = pad

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        signal_data = result['data']
        sfreq = result['freq']
        filtered_signal = mne.filter.notch_filter(
            x=signal_data,
            Fs=sfreq,
            freqs=self.freqs,
            filter_length=self.filter_length,
            notch_widths=self.notch_widths,
            trans_bandwidth=self.trans_bandwidth,
            method=self.method,
            iir_params=self.iir_params,
            mt_bandwidth=self.mt_bandwidth,
            p_value=self.p_value,
            phase=self.phase,
            fir_window=self.fir_window,
            fir_design=self.fir_design,
            pad=self.pad
        )
        result['data'] = filtered_signal
        return result

class Cheby2Filter(BaseTransform):
    def __init__(self, 
                 lowcut, 
                 highcut,  
                 order=6, 
                 rp=0.1, 
                 rs=60, 
                 btype='bandpass',
                 source: Optional[str] = None,
                 target: Optional[str] = None):
        super().__init__(source, target)
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.rp = rp
        self.rs = rs
        self.btype = btype

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        signal_data = result['data']
        sfreq = result['freq']
        nyquist = 0.5 * sfreq
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = cheby2(self.order, self.rs, [low, high], btype=self.btype)
        filtered_signal = filtfilt(b, a, signal_data, axis=1)
        result['data'] = filtered_signal
        return result