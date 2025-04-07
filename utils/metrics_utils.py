#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : metrics_utils.py
@Time    : 2024/11/15 20:28:51
@Desc    : Utility class for dynamically building and evaluating metrics.
"""

import numpy as np
import torch
from .import_utils import lazy_import_module


class MetricEvaluator:
    """
    A utility class for dynamically building and evaluating metrics.

    This class provides the following functionalities:
    1. Dynamically build and instantiate metric classes based on a given list of metric names.
    2. Store batch-wise prediction results for later evaluation.
    3. Compute and return the results of the metrics.

    Args:
        metric_list (list): A list of metric names to be dynamically instantiated.
    """

    def __init__(self, metric_list: list):
        """
        Initialize the MetricEvaluator class and build metric instances.

        Args:
            metric_list (list): A list of metric names to be dynamically instantiated.
        """
        self.metrics = self.build_metrics(metric_list)  # Build metric instances
        self.results = []  # Store batch-wise results

    def build_metrics(self, metric_list: list):
        """
        Dynamically build and return a list of metric class instances.

        Args:
            metric_list (list): A list of metric names.

        Returns:
            list: A list of instantiated metric objects.
        """
        metrics = []
        for metric_name in metric_list:
            # Dynamically import the metric class
            metric_cls = lazy_import_module("metrics", metric_name)
            # Instantiate the metric class
            metric = metric_cls()
            metrics.append(metric)
        return metrics

    def update_metrics(self, result: dict):
        """
        Store batch-wise prediction results, ensuring they are moved to the CPU.

        Args:
            result (dict): A dictionary containing batch-wise prediction results, 
                           such as 'label' and 'output'.
        """
        # Move batch results to CPU and convert to numpy arrays
        for key, value in result.items():
            # print(f"Key: {key}, Type: {type(value)}")
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                # print("Warning: Numpy array detected in result. Ensure it is on CPU.")
                result[key] = value

        self.results.append(result)

    def calculate_metrics(self):
        """
        Compute all updated metrics and return the results.

        Returns:
            dict: A dictionary containing metric names as keys and their computed results as values.
        """
        metrics_result = {}

        for metric in self.metrics:
            # Use the metric instance's name as the dictionary key
            metric_name = metric.name

            # Compute the metric result using the stored results
            result = metric.compute(self.results)

            # Convert numpy types to standard Python float
            if isinstance(result, (np.float32, np.float64)):
                result = float(result)

            # Round the result to 4 decimal places
            result = round(result, 4)

            # Add the computed result to the metrics dictionary
            metrics_result[metric_name] = result

        # Clear stored results after computation
        self.results.clear()

        return metrics_result


if __name__ == "__main__":
    # 构造测试数据 (batch, 输出层数量, 输出结果)
    true_values = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # shape: (2, 2, 2)

    predicted_values = torch.tensor([
        [[1.1, 1.9], [3.2, 3.8]],
        [[4.9, 6.1], [7.1, 7.9]]
    ])  # shape: (2, 2, 2)

    # 将测试数据包装成结果字典
    result = {"label": true_values, "output": predicted_values}

    # 初始化 MetricEvaluator，传入需要测试的指标名称
    evaluator = MetricEvaluator(metric_list=["mse", "mae"])

    # 更新指标，模拟多次 batch 的结果
    for _ in range(3):  # 假设有 3 个 batch
        evaluator.update_metrics(result)

    # 计算所有指标
    metrics_result = evaluator.calculate_metrics()

    # 打印结果
    print("Metrics Result:", metrics_result)