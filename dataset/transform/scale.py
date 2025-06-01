#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : scale.py
@Time    : 2025/03/20 11:02:05
@Desc    : 
"""

import numpy as np
from typing import Dict, Any, Optional
from dataset.transform import BaseTransform

class Scale(BaseTransform):
    def __init__(self, scale_factor: float = 1.0, source: Optional[str] = None, target: Optional[str] = None):
        """
        初始化放缩变换类。

        参数:
        - scale_factor: 放缩因子，用于对信号进行放缩。
        - source: 输入信号字段名。
        - target: 输出信号字段名。
        """
        super().__init__(source, target)
        self.scale_factor = scale_factor

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行数值放缩处理。

        参数:
        - result: 包含信号数据的字典，字段为 'data'。

        返回:
        - 更新后的信号数据字典。
        """
        data = result['data']
        result['data'] = data * self.scale_factor
        return result

class Offset(BaseTransform):
    def __init__(self, offset: float | int = 0.0, source: Optional[str] = None, target: Optional[str] = None):
        """
        初始化偏移变换类。

        参数:
        - offset: 偏移量，用于对信号进行偏移。
        - source: 输入信号字段名。
        - target: 输出信号字段名。
        """
        super().__init__(source, target)
        self.offset = offset

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行数值偏移处理。

        参数:
        - result: 包含信号数据的字典，字段为 'data'。

        返回:
        - 更新后的信号数据字典。
        """
        data = result['data']
        result['data'] = data + self.offset
        return result

class Round(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        """
        初始化四舍五入变换类。

        参数:
        - source: 输入信号字段名。
        - target: 输出信号字段名。
        """
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行四舍五入处理。

        参数:
        - result: 包含信号数据的字典，字段为 'data'。

        返回:
        - 更新后的信号数据字典。
        """
        data = result['data']
        result['data'] = np.round(data)
        return result
    
class Log(BaseTransform):
    def __init__(self, epsilon:float=1e-10, source: Optional[str] = None, target: Optional[str] = None):
        """
        初始化对数变换类。

        参数:
        - source: 输入信号字段名。
        - target: 输出信号字段名。
        """
        super().__init__(source, target)
        self.epsilon = epsilon

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行对数处理。

        参数:
        - result: 包含信号数据的字典，字段为 'data'。

        返回:
        - 更新后的信号数据字典。
        """
        data = result['data']
        result['data'] = np.log(data + self.epsilon)  # 避免对数零点
        return result