#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : metrics_utils.py
@Time    : 2024/11/15 20:28:51
@Desc    : 
"""

import numpy as np
from .import_utils import lazy_import_module


class MetricEvaluator:
    """
    MetricEvaluator 类用于根据配置字典动态构建并计算指标。

    它提供了两个主要功能：
    1. 根据给定的配置字典，动态构建和实例化各个指标类。
    2. 计算这些指标并返回结果。

    :param metric_dict: dict, 配置字典，包含指标名称和对应的参数。
    """

    def __init__(self, metric_dict: dict):
        """
        初始化 MetricEvaluator 类，构建指标实例列表。

        :param metric_dict: dict, 配置字典，其中包含指标名称及其对应的参数。
        """
        self.metrics = self.build_metrics(metric_dict)

    def build_metrics(self, metric_dict: dict):
        """
        根据配置字典动态构建并返回指标类实例列表。

        :param metric_dict: dict, 配置字典，包含指标名称及其对应的参数。
                            字典的键是指标函数的名称（例如 'accuracy'），值是该函数需要的参数（字典形式）。
                            示例：{'accuracy': {}, 'precision': {'average': 'micro'}}。
        :return: list, 指标类实例的列表，包含所有根据配置文件实例化的指标对象。
        """
        metrics = []
        # 遍历指标字典，动态导入相应的指标类并计算
        for metric_name, params in metric_dict.items():
            # 动态导入metrics中的指标类
            metric_cls = lazy_import_module("metrics", metric_name)

            # 实例化指标类对象
            metric = metric_cls(**params) if params else metric_cls()

            # 将指标对象添加到指标列表
            metrics.append(metric)
        return metrics

    def update_metrics(self, result: dict):
        """
        更新每个指标的计算数据，通常是每个 batch 的预测结果。

        :param result: dict, 包含一个 batch 的预测结果，可以包含多个字段，如 'target', 'output' 等。
        """
        target = result.get('target')
        output = result.get('output')

        if target is None or output is None:
            raise ValueError("Result dictionary must contain 'target' and 'output'.")

        for metric in self.metrics:
            # 确保 target 和 output 数据一致性
            if len(target) != len(output):
                raise ValueError("Target and output must have the same length.")
            metric.update(result)


    def calculate_metrics(self):
        """
        计算所有已更新的指标，并返回指标计算结果。

        :return: dict, 包含每个指标名称和对应计算结果的字典。
        """
        metrics_result = {}

        for metric in self.metrics:
            # 获取指标实例的类名称作为字典的键
            metric_name = metric.__class__.__name__

            # 调用每个指标实例的 compute 方法计算指标结果
            result = metric.compute()

            # 如果结果是 numpy 类型，转换为标准的 Python 浮点数
            if isinstance(result, (np.float32, np.float64)):
                result = float(result)

            # 保留结果的小数点后四位
            result = round(result, 4)

            # 将计算结果添加到结果字典
            metrics_result[metric_name] = result

            # 清空每个指标的内部结果
            metric.clear()

        return metrics_result
