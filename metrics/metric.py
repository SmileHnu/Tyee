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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    mean_squared_error, r2_score, cohen_kappa_score
)


def process_result(result: dict, is_classification: bool = True):
    """
    处理输入的 result 字典：将 Tensor 数据转换为 NumPy 数组，并根据任务类型处理输出。
    :param result: 字典，包含 'label' 和 'output'
    :param is_classification: 布尔值，判断当前任务是否为分类任务（默认是分类任务）
    :return: 处理后的字典
    """
    processed_result = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()  # 分离计算图，转为 NumPy

        if key == "output" and value is not None:
            if is_classification:
                if value.ndim == 2 and value.shape[1] > 1:  # 多分类问题
                    processed_result["output_raw"] = value.copy()  # 保留原始输出
                    value = np.argmax(value, axis=-1)  # 转为类别索引
                elif value.ndim == 2 and value.shape[1] == 1:  # 二分类问题（输出是概率值）
                    processed_result["output_raw"] = value.copy()
                    value = (value > 0.5).astype(int)  # 转为二分类标签
                elif value.ndim == 1:  # 二分类问题（输出是一维数组，通常是概率值）
                    processed_result["output_raw"] = value.copy()
                    value = (value > 0.5).astype(int)  # 转为二分类标签
                else:
                    raise ValueError(f"Unexpected output shape: {value.shape}")


        processed_result[key] = value

    return processed_result


def merge_results(accumulated_results):
    """
    合并累积的 label 和 output 数据
    """
    all_targets = np.concatenate([res[0] for res in accumulated_results], axis=0)
    all_outputs = np.concatenate([res[1] for res in accumulated_results], axis=0)
    return all_targets, all_outputs


class Metric(ABC):
    """
    指标的抽象基类，定义了 update、compute 和 reset 接口。
    """

    def __init__(self, average="binary"):
        self.average = average
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
        result = process_result(result)
        label = result.get("label")
        output = result.get("output")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return accuracy_score(all_targets, all_outputs)


class BalancedAccuracy(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return balanced_accuracy_score(all_targets, all_outputs)


class PR_AUC(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output_raw")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        if len(np.unique(all_targets)) == 1:  # 所有标签相同
            return 0.0
        precision, recall, _ = precision_recall_curve(all_targets, all_outputs)
        return auc(recall, precision)


class ROC_AUC(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output_raw")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        if len(np.unique(all_targets)) == 1:  # 所有标签相同
            return 0.0
        return roc_auc_score(all_targets, all_outputs)


class Precision(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return precision_score(all_targets, all_outputs, average=self.average)


class Recall(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return recall_score(all_targets, all_outputs, average=self.average)


class F1Score(Metric):
    def update(self, result: dict):
        result = process_result(result)
        label = result.get("label")
        output = result.get("output")
        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return f1_score(all_targets, all_outputs, average=self.average)
    
class CohenKappa(Metric):
    def __init__(self):
        super().__init__()

    def update(self, result: dict):
        """
        更新 Cohen's Kappa 指标的计算数据
        :param result: 字典，包含 'label' 和 'output'
        """
        result = process_result(result)
        label = result.get('label')
        output = result.get('output')

        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        """
        计算 Cohen's Kappa 值
        :return: Cohen's Kappa
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return cohen_kappa_score(all_targets, all_outputs)

"""   回归任务  """
class PearsonCorrelation(Metric):
    def __init__(self):
        super().__init__()

    def update(self, result: dict):
        """
        更新Pearson相关系数指标的计算数据
        :param result: 字典，包含 'label' 和 'output'
        """
        result = process_result(result, is_classification=False)
        label = result.get('label')
        output = result.get('output')

        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        """
        计算Pearson相关系数
        :return: Pearson相关系数
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return np.corrcoef(all_targets, all_outputs)[0, 1]
    

class R2Score(Metric):
    def __init__(self):
        super().__init__()

    def update(self, result: dict):
        """
        更新R²得分指标的计算数据
        :param result: 字典，包含 'label' 和 'output'
        """
        result = process_result(result, is_classification=False)
        label = result.get('label')
        output = result.get('output')

        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        """
        计算R²得分
        :return: R²得分
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        return r2_score(all_targets, all_outputs)


class RMSE(Metric):
    def __init__(self):
        super().__init__()

    def update(self, result: dict):
        """
        更新RMSE指标的计算数据
        :param result: 字典，包含 'label' 和 'output'
        """
        result = process_result(result, is_classification=False)
        label = result.get('label')
        output = result.get('output')

        if label is not None and output is not None:
            self._accumulated_results.append((label, output))

    def compute(self):
        """
        计算均方根误差(RMSE)
        :return: RMSE值
        """
        all_targets, all_outputs = merge_results(self._accumulated_results)
        mse = mean_squared_error(all_targets, all_outputs)
        return np.sqrt(mse)