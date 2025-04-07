#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : wavelet.py
@Time    : 2025/03/29 20:53:09
@Desc    : 
"""
import pywt
import numpy as np
from utils import lazy_import_module
from .base_transform import BaseTransform

class WaveletTransform(BaseTransform):
    def __init__(self, wavelet = 'db5', level = 5, **kwargs):
        """
        初始化清洗变换类。

        参数:
        - type_clean: 清洗信号类型 ecg_clean, emg_clean等，见neurokit2的clean方法。
        - method: 清洗方法，'neurokit' ,参考neurokit2的具体clean方法。
        - kwargs: 清洗方法的参数。
        """
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.kwargs = kwargs
    
    def transform(self, result):
        """
        对信号进行清洗处理。

        参数:
        - result: 包含信号数据的字典。

        返回:
        - 更新后的信号数据字典。
        """
        signals = result['signals']  # 假设 signals 的 shape 是 (通道数, 采样点数)
        processed_signals = []

        # 遍历每个通道
        for channel_data in signals:
            # 小波变换
            coeffs = pywt.wavedec(data=channel_data, wavelet=self.wavelet, level=self.level)
            
            # 阈值去噪
            cD1 = coeffs[-1]  # 最细一级的细节系数
            threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
            coeffs[1:] = [pywt.threshold(c, threshold) for c in coeffs[1:]]  # 对细节系数进行阈值处理

            # 小波反变换，获取去噪后的信号
            rdata = pywt.waverec(coeffs=coeffs, wavelet=self.wavelet)
            processed_signals.append(rdata)

        # 将处理后的信号重新组合为 (通道数, 采样点数) 的形状
        result['signals'] = np.array(processed_signals)
        return result
