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
from typing import Dict, Any
from dataset.transform import BaseTransform

class Scale(BaseTransform):
    def __init__(self, scale_factor: float = 1.0):
        """
        初始化放缩变换类。

        参数:
        - scale_factor: 放缩因子，用于对信号进行放缩。
        """
        super().__init__()
        self.scale_factor = scale_factor

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        对信号进行数值放缩处理。

        参数:
        - result: 包含信号数据的字典。

        返回:
        - 更新后的信号数据字典。
        """
        signals = result['signals']
        result['signals'] = signals * self.scale_factor
        # print('Scale transform')
        return result