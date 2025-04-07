#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : car.py
@Time    : 2025/03/30 20:35:06
@Desc    : 
"""


from typing import Dict, Any
import numpy as np
from .base_transform import BaseTransform

class CAR(BaseTransform):
    def __init__(self):
        """
        初始化 CAR 类，用于计算公共平均参考 (Common Average Referencing, CAR)。
        """
        pass

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对输入信号进行公共平均参考 (CAR) 转换。

        参数:
            result (Dict[str, Any]): 包含信号数据的字典，必须包含 'signals' 键。

        返回:
            Dict[str, Any]: 包含经过 CAR 转换的信号的字典。
        """
        # 获取信号数据
        signals = result.get("signals")
        if signals is None:
            raise ValueError("输入字典中必须包含 'signals' 键")

        # 检查信号维度是否为 2D 或 3D
        if signals.ndim == 2:
            # 二维信号 (n_channels, n_times)
            common_average = np.mean(signals, axis=0, keepdims=True)  # 计算每个时间点的平均值
            signals = signals - common_average  # 每个通道减去公共平均值
        elif signals.ndim == 3:
            # 三维信号 (n_epochs, n_channels, n_times)
            common_average = np.mean(signals, axis=1, keepdims=True)  # 对每个 epoch 的通道取平均
            signals = signals - common_average  # 每个通道减去公共平均值
        else:
            raise ValueError(f"不支持的信号维度: {signals.ndim}D，支持 2D 或 3D 信号")

        # 更新结果字典
        result["signals"] = signals
        return result