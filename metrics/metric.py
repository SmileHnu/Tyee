#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : metric.py
@Time    : 2024/11/15 20:28:38
@Desc    : 
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def process_result(result: dict):
    """
    处理输入的 result 字典：
    1. 将其中的 Tensor 数据转换为 CPU 上的 NumPy 数组。
    2. 如果 output 是概率分布（多维数组），转换为类别标签。

    :param result: 字典，包含 'target' 和 'output'。
    :return: 处理后的字典。
    """
    processed_result = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()  # 确保与计算图分离

        # 如果是 output 且是概率分布，将其转换为类别标签
        if key == 'output' and value is not None:
            if value.ndim > 1:
                value = np.argmax(value, axis=1)  # 多分类转换为类别索引
            elif value.ndim == 1:
                pass  # 一维数据，无需处理
            else:
                raise ValueError(f"Unexpected output shape: {value.shape}")

        processed_result[key] = value

    return processed_result


def merge_results(accumulated_results):
    """
    合并累积的 target 和 output 数据
    :param accumulated_results: 累积的结果列表 [(target1, output1), (target2, output2), ...]
    :return: all_targets, all_outputs 合并后的 NumPy 数组
    """
    all_targets = np.array([])
    all_outputs = np.array([])
    for target, output in accumulated_results:
        all_targets = np.concatenate((all_targets, target), axis=0) if all_targets.size > 0 else target
        all_outputs = np.concatenate((all_outputs, output), axis=0) if all_outputs.size > 0 else output
    return all_targets, all_outputs


class Metric(ABC):
    """
    指标的父类，定义了计算指标的接口
    """
    def __init__(self):
        self.reset()

    @abstractmethod
    def update(self, result: dict):
        pass

    @abstractmethod
    def compute(self):
        pass

    def reset(self):
        self._accumulated_results = []

    def clear(self):
        self.reset()


class Accuracy(Metric):
    def update(self, result: dict):
        """
        更新准确率指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """

        result = process_result(result)
        target = result.get('target')
        output = result.get('output')

        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算准确率
        :return: 准确率
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return accuracy_score(all_targets, all_outputs)


class Precision(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新精确度指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        result = process_result(result)
        target = result.get('target')
        output = result.get('output')

        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算精确度
        :return: 精确度
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return precision_score(all_targets, all_outputs, average=self.average)


class Recall(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新召回率指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        result = process_result(result)
        target = result.get('target')
        output = result.get('output')

        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算召回率
        :return: 召回率
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return recall_score(all_targets, all_outputs, average=self.average)


class F1(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新F1分数指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        result = process_result(result)
        target = result.get('target')
        output = result.get('output')

        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算F1分数
        :return: F1分数
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return f1_score(all_targets, all_outputs, average=self.average)

