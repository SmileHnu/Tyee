#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : regression_metrics.py
@Time    : 2025/01/05 17:23:17
@Desc    : 
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
class RegressionMetric(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
    
    def process_result(self, results: list):
        """
        处理输入的结果列表：将 Tensor 数据转换为 NumPy 数组，并根据任务类型处理输出。
        :param results: 列表，每个元素是一个字典，包含 'loss'、'target' 和 'output' 等元素。
        :return: 元组，包含处理后的 target 和 output 列表。
        """
        all_targets = []
        all_outputs = []

        for result in results:
            target = result.get('target')
            output = result.get('output')
            target = target.flatten()
            output = output.flatten()
            if target is not None and output is not None:
                all_targets.append(target)
                all_outputs.append(output)
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_targets, all_outputs
    @abstractmethod
    def compute(self, results: list):
        """
        计算指标。
        :param results: 列表，每个元素是一个字典，包含 'loss'、'target' 和 'output' 等元素。
        :return: 字典，包含指标名称和值。
        """
        raise NotImplementedError
    
class MeanSquaredError(RegressionMetric):
    def __init__(self):
        self.name = 'mse'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return mean_squared_error(all_targets, all_outputs)

class R2Score(RegressionMetric):
    def __init__(self):
        self.name = 'r2'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return r2_score(all_targets, all_outputs)

class MeanAbsoluteError(RegressionMetric):
    def __init__(self):
        self.name = 'mae'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return mean_absolute_error(all_targets, all_outputs)
        
class RMSE(RegressionMetric):
    def __init__(self):
        self.name = 'rmse'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return np.sqrt(mean_squared_error(all_targets, all_outputs))