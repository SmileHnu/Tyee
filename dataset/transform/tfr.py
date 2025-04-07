#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : tfr.py
@Time    : 2025/03/30 17:30:52
@Desc    : 
"""

from typing import Dict, Any
import numpy as np
from .base_transform import BaseTransform
from mne.time_frequency import tfr_array_morlet

class TFR(BaseTransform):
    def __init__(
        self,
        freqs: np.ndarray,
        n_cycles: float = 7.0,
        zero_mean: bool = True,
        use_fft: bool = True,
        decim: int = 1,
        output: str = "complex",
        n_jobs: int = None,
        verbose = None,
    ):
        """
        初始化 TFRTransform 类，用于计算时间-频率表示 (TFR)。

        参数:
            sfreq (float): 数据的采样频率。
            freqs (np.ndarray): 要分析的频率数组。
            n_cycles (float): 每个频率的周期数，默认为 7.0。
            zero_mean (bool): 是否使小波均值为零，默认为 True。
            use_fft (bool): 是否使用 FFT 进行卷积，默认为 True。
            decim (int): 下采样因子，默认为 1。
            output (str): 输出类型，默认为 "complex"。
            n_jobs (int): 并行处理的线程数，默认为 None。
        """
        self.freqs = freqs
        self.n_cycles = n_cycles
        self.zero_mean = zero_mean
        self.use_fft = use_fft
        self.decim = decim
        self.output = output
        self.n_jobs = n_jobs
        self.verbose = verbose

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对输入数据计算时间-频率表示 (TFR)。

        参数:
            result (Dict[str, Any]): 包含信号数据的字典，必须包含 "signals" 键。

        返回:
            Dict[str, Any]: 包含 TFR 结果的字典。
        """
        if "signals" not in result:
            raise ValueError("输入字典中必须包含 'signals' 键")

        data = result["signals"]
        sfreq = result['sampling_rate']
        if not isinstance(data, np.ndarray):
            raise TypeError("'signals' 必须是一个 NumPy 数组")
        num_of_channels = len(data)
        data = data.reshape(1, num_of_channels, -1)

        # 调用 tfr_array_morlet 计算 TFR
        tfr_result = tfr_array_morlet(
            data=data,
            sfreq=sfreq,
            freqs=self.freqs,
            n_cycles=self.n_cycles,
            zero_mean=self.zero_mean,
            use_fft=self.use_fft,
            decim=self.decim,
            output=self.output,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        # 将结果存入字典并返回
        result["signals"] = tfr_result
        return result