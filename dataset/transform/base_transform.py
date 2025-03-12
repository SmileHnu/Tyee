#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : base_transform.py
@Time    : 2025/02/23 20:44:09
@Desc    : 
"""

from typing import Dict, Any

class BaseTransform:
    def __call__(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用 transform 函数对 result 进行处理。

        参数:
            result (Dict[str, Any]): 包含信号数据、通道和标签的字典。

        返回:
            Dict[str, Any]: 处理后的 result 字典。
        """
        return self.transform(result)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        子类需要实现的具体预处理逻辑。

        参数:
            result (Dict[str, Any]): 包含信号数据、通道和标签的字典。

        返回:
            Dict[str, Any]: 处理后的 result 字典。
        """
        raise NotImplementedError("子类需要实现这个方法")