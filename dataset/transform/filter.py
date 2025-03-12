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
from typing import Dict, Any, List
from dataset.transform import BaseTransform
from neurokit2 import signal_filter

class NeurokitFilter(BaseTransform):
    def __init__(self, 
                 lowcut=None,
                 highcut=None,
                 method="butterworth",
                 order=2,
                 window_size="default",
                 powerline=50,
                 show=False,):
        super().__init__()
        self.lowcut = lowcut
        self.highcut = highcut
        self.method = method
        self.order = order
        self.window_size = window_size
        self.powerline = powerline
        self.show = show
    
    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        
        result['signals'] = signal_filter(result['signals'], 
                                       result['sampling_rate'], 
                                       self.lowcut, 
                                       self.highcut, 
                                       self.method, 
                                       self.order, 
                                       self.window_size, 
                                       self.powerline, 
                                       self.show)
        return result

class Filter(BaseTransform):
    def __init__(self, 
                 l_freq, 
                 h_freq, 
                 filter_length="auto", 
                 l_trans_bandwidth="auto", 
                 h_trans_bandwidth="auto", 
                 method="fir", 
                 iir_params=None, 
                 phase="zero", 
                 fir_window="hamming", 
                 fir_design="firwin", 
                 pad="reflect_limited"):
        super().__init__()
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
        # print('执行了Filter')
        """
        对信号进行滤波。
        
        参数:
        - result: 包含信号数据的字典。
        
        返回:
        - 更新后的信号数据字典。
        """
        
        signal_data = result['signals']
        sfreq = result['sampling_rate']
        
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
        # print(filtered_signal)
        result['signals'] = filtered_signal
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
                 pad="reflect_limited"):
        super().__init__()
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
        # print('执行了NotchFilter')
        """
        对信号进行陷波滤波。
        
        参数:
        - result: 包含信号数据的字典。
        
        返回:
        - 更新后的信号数据字典。
        """
        signal_data = result['signals']
        sfreq = result['sampling_rate']

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
        # print(filtered_signal)
        result['signals'] = filtered_signal
        return result
