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
import scipy.ndimage
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
        # print(all_outputs, all_targets)
        return all_targets, all_outputs
    @abstractmethod
    def compute(self, results: list):
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
        all_targets, all_outputs = self.process_result(results)
        cos_sim = cosine_similarity(all_targets, all_outputs)
        return np.mean(cos_sim)

class PearsonCorrelation(RegressionMetric):
    def __init__(self):
        self.name = 'pearson_correlation'

    def compute(self, results: list):
        all_targets, all_outputs = self.process_result(results)
        r = np.corrcoef(all_targets, all_outputs)[0, 1]
        return r

class MeanCC(RegressionMetric):
    def __init__(self):
        self.name = 'mean_cc'

    def compute(self, results: list):
        all_targets = []
        all_outputs = []
        for result in results:
            label = result.get('label')
            output = result.get('output')
            if label is not None and output is not None:
                all_targets.append(label)
                all_outputs.append(output)
        all_targets = np.concatenate(all_targets, axis=0)  
        all_outputs = np.concatenate(all_outputs, axis=0)

        all_targets = np.concatenate(all_targets, axis=-1)
        all_outputs = np.concatenate(all_outputs, axis=-1)

        all_outputs = scipy.ndimage.gaussian_filter1d(all_outputs, sigma=6, axis=-1)
        corrs = []
        for i in range(all_targets.shape[0]):
            corr = np.corrcoef(all_outputs[i], all_targets[i])[0, 1]
            corrs.append(corr)
        return np.mean(corrs)
