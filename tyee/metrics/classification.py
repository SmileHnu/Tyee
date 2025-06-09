#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : classification.py
@Time    : 2025/01/06 14:35:26
@Desc    : 
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, average_precision_score,
    jaccard_score, roc_auc_score, precision_score, recall_score,
    f1_score, cohen_kappa_score
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ClassMetric(ABC):
    """
    指标的抽象基类，定义了 compute 接口。
    """

    def __init__(self, average="binary"):
        self.average = average
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

            # Handle multi-class and binary classification outputs
            if output.ndim == 2 and output.shape[1] > 1:  # Multi-class classification
                output = np.argmax(output, axis=-1)
            elif output.ndim == 2 and output.shape[1] == 1:  # Binary classification (probabilities)
                output = (output > 0.5).astype(int)
            elif output.ndim == 1:  # Binary classification (1D probabilities)
                output = (output > 0.5).astype(int)

            if label is not None and output is not None:
                all_targets.append(label)
                all_outputs.append(output)

        # Concatenate all targets and outputs
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        return all_targets, all_outputs
    
    def process_roc_auc_results(self, results: list):
        all_targets_roc = []
        all_outputs_probs = [] 

        for result in results:
            label = result.get('label') 
            output_raw = result.get('output') 

            if label.ndim == 2 and label.shape[1] > 1:
                if np.array_equal(label, label.astype(bool)):
                    label = np.argmax(label, axis=1)
            
            if label is not None and output_raw is not None:
                all_targets_roc.append(label)
                all_outputs_probs.append(output_raw)

        if not all_targets_roc or not all_outputs_probs:
            return np.array([]), np.array([])


        all_targets_roc = np.concatenate(all_targets_roc, axis=0)
        all_outputs_probs = np.concatenate(all_outputs_probs, axis=0)
        
        return all_targets_roc, all_outputs_probs
    
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
class TruePositive(ClassMetric):
    """计算真阳性(TP)"""
    def __init__(self):
        self.name = 'tp'
    
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        tp = int(np.sum((all_targets == 1) & (all_outputs == 1)))
        return tp

class TrueNegative(ClassMetric):
    """计算真阴性(TN)"""
    def __init__(self):
        self.name = 'tn'
    
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        tn = int(np.sum((all_targets == 0) & (all_outputs == 0)))
        return tn

class FalsePositive(ClassMetric):
    """计算假阳性(FP)"""
    def __init__(self):
        self.name = 'fp'
    
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        fp = int(np.sum((all_targets == 0) & (all_outputs == 1)))
        return fp

class FalseNegative(ClassMetric):
    """计算假阴性(FN)"""
    def __init__(self):
        self.name = 'fn'
    
    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        fn = int(np.sum((all_targets == 1) & (all_outputs == 0)))
        return fn
    
class PR_AUC(ClassMetric):
    def __init__(self):
        self.name = 'pr_auc'
    def compute(self, results: list):
        all_targets = []
        all_probs = []
        for result in results:
            label = result.get('label')
            output = result.get('output')
            # 取概率分数
            if output.ndim == 2 and output.shape[1] == 1:
                prob = output[:, 0]
            elif output.ndim == 2 and output.shape[1] > 1:
                # 假设正类为1
                prob = output[:, 1]
            else:
                prob = output
            all_targets.append(label)
            all_probs.append(prob)
        all_targets = np.concatenate(all_targets, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        return average_precision_score(all_targets, all_probs)

class ROC_AUC(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc'
    def compute(self, results: list):
        all_targets = []
        all_probs = []
        for result in results:
            label = result.get('label')
            output = result.get('output')
            if output.ndim == 2 and output.shape[1] == 1:
                prob = output[:, 0]
            elif output.ndim == 2 and output.shape[1] > 1:
                prob = output[:, 1]
            else:
                prob = output
            all_targets.append(label)
            all_probs.append(prob)
        all_targets = np.concatenate(all_targets, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        return roc_auc_score(all_targets, all_probs)

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
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovo')

class ROC_AUC_Macro_OVR(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_macro_ovr'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return roc_auc_score(all_targets, all_outputs, average='macro', multi_class='ovr')

class ROC_AUC_Weighted_OVO(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_weighted_ovo'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovo')

class ROC_AUC_Weighted_OVR(ClassMetric):
    def __init__(self):
        self.name = 'roc_auc_weighted_ovr'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return roc_auc_score(all_targets, all_outputs, average='weighted', multi_class='ovr')

class PR_AUC_Macro(ClassMetric):
    def __init__(self):
        self.name = 'pr_auc_macro'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return average_precision_score(all_targets, all_outputs, average='macro')

class PR_AUC_Weighted(ClassMetric):
    def __init__(self):
        self.name = 'pr_auc_weighted'
    def compute(self, results: list):
        all_targets, all_outputs = self.process_roc_auc_results(results)
        return average_precision_score(all_targets, all_outputs, average='weighted')


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

