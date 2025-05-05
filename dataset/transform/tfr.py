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

from typing import Dict, Any, Optional
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
        n_jobs: Optional[int] = None,
        verbose: Optional[bool] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ):
        """
        初始化 TFR 类，用于计算时间-频率表示 (TFR)。

        参数:
            freqs (np.ndarray): 要分析的频率数组。
            n_cycles (float): 每个频率的周期数，默认为 7.0。
            zero_mean (bool): 是否使小波均值为零，默认为 True。
            use_fft (bool): 是否使用 FFT 进行卷积，默认为 True。
            decim (int): 下采样因子，默认为 1。
            output (str): 输出类型，默认为 "complex"。
            n_jobs (int): 并行处理的线程数，默认为 None。
            source (str): 输入信号字段名，默认为 'data'。
            target (str): 输出信号字段名，默认为 None（覆盖输入）。
        """
        super().__init__(source, target)
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
            result (Dict[str, Any]): 包含信号数据的字典，字段为 'data' 和 'freq'。

        返回:
            Dict[str, Any]: 包含 TFR 结果的字典。
        """
        if "data" not in result or "freq" not in result:
            raise ValueError("输入字典中必须包含 'data' 和 'freq' 字段")

        data = result["data"]
        sfreq = result["freq"]
        if not isinstance(data, np.ndarray):
            raise TypeError("'data' 必须是一个 NumPy 数组")
        if data.ndim == 2:
            # (channels, times) -> (1, channels, times)
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            # (epochs, channels, times) 保持原样
            pass
        else:
            raise ValueError("输入 'data' 必须为2维或3维数组，当前shape: {}".format(data.shape))

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

        # 输出 shape: (epochs, channels, freqs, times)
        result["data"] = tfr_result
        result["freqs"] = self.freqs
        return result