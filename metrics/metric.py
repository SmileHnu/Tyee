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

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metric(ABC):
    """
    指标的父类，定义了计算指标的接口
    """

    def __init__(self):
        self.reset()

    @abstractmethod
    def update(self, result: dict):
        """
        更新指标的计算数据，通常是将每个 batch 的预测值和真实值累积到一个内部结构中。
        :param result: 一个包含多个键值对的字典，可能包括 'target', 'output'，或者其他你需要的字段
        """
        pass

    @abstractmethod
    def compute(self):
        """
        计算并返回当前累计的指标值
        :return: 当前的指标值
        """
        pass

    def reset(self):
        """
        重置指标的内部状态，用于清除历史数据
        """
        self._accumulated_results = []

    def clear(self):
        """
        清除指标的内部状态，通常在计算完指标后调用
        """
        self.reset()


# 子类：准确率指标
class Accuracy(Metric):
    def update(self, result: dict):
        """
        更新准确率指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        target = result.get('target')
        output = result.get('output')
        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算准确率
        :return: 准确率
        """
        all_targets = []
        all_outputs = []
        for target, output in self._accumulated_results:
            all_targets.extend(target)
            all_outputs.extend(output)
        return accuracy_score(all_targets, all_outputs)


# 子类：精确度指标
class Precision(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新精确度指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        target = result.get('target')
        output = result.get('output')
        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算精确度
        :return: 精确度
        """
        all_targets = []
        all_outputs = []
        for target, output in self._accumulated_results:
            all_targets.extend(target)
            all_outputs.extend(output)
        return precision_score(all_targets, all_outputs, average=self.average)


# 子类：召回率指标
class Recall(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新召回率指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        target = result.get('target')
        output = result.get('output')
        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算召回率
        :return: 召回率
        """
        all_targets = []
        all_outputs = []
        for target, output in self._accumulated_results:
            all_targets.extend(target)
            all_outputs.extend(output)
        return recall_score(all_targets, all_outputs, average=self.average)


# 子类：F1分数指标
class F1(Metric):
    def __init__(self, average='binary'):
        super().__init__()
        self.average = average

    def update(self, result: dict):
        """
        更新F1分数指标的计算数据
        :param result: 字典，包含 'target' 和 'output'
        """
        target = result.get('target')
        output = result.get('output')
        if target is not None and output is not None:
            self._accumulated_results.append((target, output))

    def compute(self):
        """
        计算F1分数
        :return: F1分数
        """
        all_targets = []
        all_outputs = []
        for target, output in self._accumulated_results:
            all_targets.extend(target)
            all_outputs.extend(output)
        return f1_score(all_targets, all_outputs, average=self.average)
