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
