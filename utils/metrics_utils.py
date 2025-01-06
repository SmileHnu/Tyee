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
import torch
from .import_utils import lazy_import_module


class MetricEvaluator:
    """
    MetricEvaluator 类用于根据配置字典动态构建并计算指标。

    它提供了两个主要功能：
    1. 根据给定的指标列表，动态构建和实例化各个指标类。
    2. 保存每个 batch 的预测结果，并在需要时计算这些指标。
    3. 计算这些指标并返回结果。

    :param metric_list: list, 包含指标名称的列表。
    """

    def __init__(self, metric_list: list):
        """
        初始化 MetricEvaluator 类，构建指标实例列表。

        :param metric_list: list, 包含指标名称的列表。
        """
        self.metrics = self.build_metrics(metric_list)  # 构建指标实例列表
        self.results = []  # 用于存储所有的 result

    def build_metrics(self, metric_list: list):
        """
        根据指标列表动态构建并返回指标类实例列表。

        :param metric_list: list, 包含指标名称的列表。
        :return: list, 指标类实例的列表，包含所有根据配置文件实例化的指标对象。
        """
        metrics = []
        for metric_name in metric_list:
            # 动态导入指标类
            metric_cls = lazy_import_module(f"metrics", metric_name)
            # 实例化指标类
            metric = metric_cls()
            metrics.append(metric)
        return metrics

    def update_metrics(self, result: dict):
        """
        保存每个 batch 的预测结果，需要转移到CPU上。

        :param result: dict, 包含一个 batch 的预测结果，可以包含多个字段，如 'target', 'output' 等。
        """
        # 将每个 batch 的结果转移到 CPU 上并转换为 numpy 数组
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.detach().cpu()
            else:
                result[key] = value.cpu()
            
        self.results.append(result)

    def calculate_metrics(self):
        """
        计算所有已更新的指标，并返回指标计算结果。

        :return: dict, 包含每个指标名称和对应计算结果的字典。
        """
        metrics_result = {}

        for metric in self.metrics:
            
            # 获取指标实例的类名称作为字典的键
            metric_name = metric.name

            # 调用每个指标实例的 compute 方法计算指标结果
            result = metric.compute(self.results)

            # 如果结果是 numpy 类型，转换为标准的 Python 浮点数
            if isinstance(result, (np.float32, np.float64)):
                result = float(result)

            # 保留结果的小数点后四位
            result = round(result, 4)

            # 将计算结果添加到结果字典
            metrics_result[metric_name] = result

        # 清空 results 列表
        self.results.clear()

        return metrics_result
