#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : class_metrics.py
@Time    : 2025/01/06 14:35:26
@Desc    : 
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score,\
                            jaccard_score, roc_auc_score, precision_score, \
                            recall_score, f1_score, cohen_kappa_score
from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ClassMetric(ABC):
    """
    指标的抽象基类，定义了 compute 接口。
    """

    def __init__(self, average="binary"):
        """
        初始化 Metric 类。

        :param average: str, 指定指标的计算方式，例如 'binary' 或其他类型。
        :param name: str, 指标的名称。
        """
        self.average = average
        self.name = self.__class__.__name__
    
    def process_result(self, results: list):
        """
        处理输入的结果列表：将 Tensor 数据转换为 NumPy 数组，并根据任务类型处理输出。
        :param results: 列表，每个元素是一个字典，包含 'loss'、'label' 和 'output' 等元素。
        :return: 元组，包含处理后的 label 和 output 列表。
        """
        all_targets = []
        all_outputs = []

        for result in results:
            label = result.get('label')
            output = result.get('output')
            if output.ndim == 2 and output.shape[1] > 1:  # 多分类问题
                output = np.argmax(output, axis=-1)
            elif output.ndim == 2 and output.shape[1] == 1:  # 二分类问题（输出是概率值）
                output = (output > 0.5).astype(int)
            elif output.ndim == 1:  # 二分类问题（输出是一维数组，通常是概率值）
                output = (output > 0.5).astype(int)
            if label is not None and output is not None:
                all_targets.append(label)
                all_outputs.append(output)
        
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        return all_targets, all_outputs
    
    @abstractmethod
    def compute(self, result: list):
        """
        计算并返回指标结果。

        :param result: dict, 包含计算指标所需的数据。
        :return: float, 计算得到的指标结果。
        """
        raise NotImplementedError
    
class Accuracy(ClassMetric):
    def __init__(self):
        self.name = 'accuracy'

    def compute(self, result: list):
        all_targets, all_outputs = self.process_result(result)
        return accuracy_score(all_targets, all_outputs)


class BalancedAccuracy(ClassMetric):
    def __init__(self):
        self.name = 'balanced_accuracy'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return balanced_accuracy_score(all_targets, all_outputs)

class CohenKappa(ClassMetric):
    def __init__(self):
        self.name = 'cohen_kappa'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return cohen_kappa_score(all_targets, all_outputs)
    
# 二分类问题的指标
class PR_AUC(ClassMetric):
    def __init__(self):
        self.name = 'pr_auc'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return average_precision_score(all_targets, all_outputs)


class ROC_AUC(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return roc_auc_score(all_targets, all_outputs)

class Precision(ClassMetric):
    def __init__(self):
        self.name = 'precision'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return precision_score(all_targets, all_outputs)

class Recall(ClassMetric):
    def __init__(self):
        self.name = 'recall'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return recall_score(all_targets, all_outputs)

class F1(ClassMetric):
    def __init__(self):
        self.name = 'f1'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return f1_score(all_targets, all_outputs)
    
class Jaccard(ClassMetric):
    def __init__(self):
        self.name = 'jaccard'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return jaccard_score(all_targets, all_outputs)

# 多分类问题的指标
class ROC_AUC_Macro_OVO(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_macro_ovo'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')

class ROC_AUC_Macro_OVR(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_macro_ovr'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovr')

class ROC_AUC_Weighted_OVO(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_weighted_ovo'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovo')

class ROC_AUC_Weighted_OVR(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_weighted_ovr'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovr')

class F1_Macro(ClassMetric):
    def __init__(self):
        self.name = 'f1_macro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return f1_score(all_targets, all_outputs, average='macro')

class F1_Micro(ClassMetric):
    def __init__(self):
        self.name = 'f1_micro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return f1_score(all_targets, all_outputs, average='micro')

class F1_Weighted(ClassMetric):
    def __init__(self):
        self.name = 'f1_weighted'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return f1_score(all_targets, all_outputs, average='weighted')
    
class Precision_Macro(ClassMetric):
    def __init__(self):
        self.name = 'precision_macro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return precision_score(all_targets, all_outputs, average='macro')

class Precision_Micro(ClassMetric):
    def __init__(self):
        self.name = 'precision_micro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return precision_score(all_targets, all_outputs, average='micro')
    
class Precision_Weighted(ClassMetric):
    def __init__(self):
        self.name = 'precision_weighted'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return precision_score(all_targets, all_outputs, average='weighted')

class Recall_Macro(ClassMetric):
    def __init__(self):
        self.name = 'recall_macro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return recall_score(all_targets, all_outputs, average='macro')

class Recall_Micro(ClassMetric):
    def __init__(self):
        self.name = 'recall_micro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return recall_score(all_targets, all_outputs, average='micro')

class Recall_Weighted(ClassMetric):
    def __init__(self):
        self.name = 'recall_weighted'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return recall_score(all_targets, all_outputs, average='weighted')
    
class jaccard_Macro(ClassMetric):
    def __init__(self):
        self.name = 'jaccard_macro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return jaccard_score(all_targets, all_outputs, average='macro')

class jaccard_Micro(ClassMetric):
    def __init__(self):
        self.name = 'jaccard_micro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return jaccard_score(all_targets, all_outputs, average='micro')
    
class jaccard_Weighted(ClassMetric):
    def __init__(self):
        self.name = 'jaccard_weighted'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        return jaccard_score(all_targets, all_outputs, average='weighted')

