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
import scipy
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
class RegressionMetric(ABC):
    def __init__(self):
        self.name = self.__class__.__name__
    
    def process_result(self, results: list):
        """
        Process the input results list: handle the output based on task type.

        Args:
            results (list): A list of dictionaries, each containing 'label' and 'output'.

        Returns:
            tuple: Processed label and output lists.
        """
        all_targets = []
        all_outputs = []

        for result in results:
            label = result.get('label')
            output = result.get('output')
            label = label.flatten()
            output = output.flatten()
            if label is not None and output is not None:
                all_targets.append(label)
                all_outputs.append(output)
        # Concatenate all targets and outputs
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        # all_outputs = all_outputs.flatten()
        # all_targets = all_targets.flatten()
        print(all_outputs, all_targets)
        return all_targets, all_outputs
    @abstractmethod
    def compute(self, results: list):
        """
        计算指标。
        :param results: 列表，每个元素是一个字典，包含 'loss'、'label' 和 'output' 等元素。
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

class CosineSimilarity(RegressionMetric):
    def __init__(self):
        self.name = 'cosine_similarity'

    def compute(self, results: list):
        """
        计算所有目标和输出之间的余弦相似度
        """
        all_targets, all_outputs = self.process_result(results)
        # 使用 sklearn 的 cosine_similarity 计算余弦相似度
        cos_sim = cosine_similarity(all_targets, all_outputs)
        return np.mean(cos_sim)

class PearsonCorrelation(RegressionMetric):
    def __init__(self):
        self.name = 'pearson_correlation'

    def compute(self, results: list):
        """
        计算所有目标和输出之间的皮尔逊相关系数
        """
        all_targets, all_outputs = self.process_result(results)
        # 使用 numpy 的 corrcoef 计算皮尔逊相关系数
        r = np.corrcoef(all_targets, all_outputs)[0, 1]
        return r

class MeanCC(RegressionMetric):
    def __init__(self):
        self.name = 'mean_cc'

    def compute(self, results: list):
        """
        计算所有时间步长上的皮尔逊相关系数的平均值
        """
        # 拼接所有 batch，得到 (总样本数, 通道数, 样本数)
        all_targets = []
        all_outputs = []
        for result in results:
            # (batch, 通道数, 样本数)
            label = result.get('label')
            output = result.get('output')
            if label is not None and output is not None:
                all_targets.append(label)
                all_outputs.append(output)
        # 拼接 batch 维
        all_targets = np.concatenate(all_targets, axis=0)  # (总batch, 通道数, 样本数)
        all_outputs = np.concatenate(all_outputs, axis=0)

        # 变成 (通道数, 总样本数)
        # 先把 batch 和 样本数 合并
        all_targets = np.concatenate(all_targets, axis=-1)  # (总batch, 通道数, 总样本数) -> (通道数, 总样本数)
        all_outputs = np.concatenate(all_outputs, axis=-1)

        corrs = []
        for i in range(all_targets.shape[0]):
            corr = np.corrcoef(all_outputs[i], all_targets[i])[0, 1]
            corrs.append(corr)
        return np.mean(corrs)

if __name__ == "__main__":
    from pyhealth.metrics import regression_metrics_fn
    # 构造测试数据 (batch, 输出层数量, 输出结果)
    true_values = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # shape: (2, 2, 2)

    predicted_values = np.array([
        [[1.1, 1.9], [3.2, 3.8]],
        [[4.9, 6.1], [7.1, 7.9]]
    ])  # shape: (2, 2, 2)

    # 将测试数据包装成结果列表
    
    result = {"label": true_values, "output": predicted_values}
    results = []
    x = []
    x_rec = []
    for i in range(3):
        results.append(result)
        x.append(true_values)
        x_rec.append(predicted_values)
    x = np.concatenate(x, axis=0)
    x_rec = np.concatenate(x_rec, axis=0)

    # 初始化指标类
    metrics = [
        MeanSquaredError(),
        # R2Score(),
        MeanAbsoluteError(),
        # RMSE(),
        # CosineSimilarity(),
        # PearsonCorrelation(),
        # MeanCC()
    ]

    # 逐个测试指标
    for metric in metrics:
        try:
            result = metric.compute(results)
            print(f"{metric.name}: {result}")
        except Exception as e:
            print(f"Error in {metric.name}: {e}")
    # 测试 pyhealth.metrics 模块
    result = regression_metrics_fn(x, x_rec, metrics=['mse','mae'])
    print(f"pyhealth.metrics: {result}")