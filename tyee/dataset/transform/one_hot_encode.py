#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : one_hot_encode.py
@Time    : 2025/05/21 20:53:39
@Desc    : 
"""
from tyee.dataset.transform import BaseTransform
import numpy as np

class OneHotEncode(BaseTransform):
    def __init__(self, num: int, source=None, target=None):
        super().__init__(source, target)
        self.num = num
    
    def transform(self, result):
        """
        Converts the input data to one-hot encoded format.

        Args:
            result (dict): Input data containing 'data' key.
                           'data' can be a scalar (int) or an array-like (list, np.ndarray).

        Returns:
            dict: One-hot encoded data.
        """
        data = result['data']

        # Check if data is a scalar (Python int, float, or 0-dim NumPy array)
        if np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0):
            try:
                idx = int(data)  # Convert to integer for indexing
            except ValueError:
                raise TypeError(f"Cannot convert scalar data '{data}' to an integer for one-hot encoding.")

            # Check if the index is within the valid range
            if not (0 <= idx < self.num):
                raise ValueError(
                    f"Scalar data {idx} is out of range [0, {self.num - 1}] for {self.num} classes."
                )
            
            one_hot = np.zeros(self.num, dtype=np.float32)
            one_hot[idx] = 1
        else:
            # Handle array-like input (list, tuple, or multi-dimensional NumPy array)
            try:
                # Convert to a 1D NumPy integer array
                data_arr = np.asarray(data).flatten().astype(int)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert input data of type {type(data)} to a 1D integer NumPy array for one-hot encoding. Error: {e}"
                )

            # If the array is not empty, check if all elements are within the valid range
            if data_arr.size > 0:
                if not (np.all(data_arr >= 0) & np.all(data_arr < self.num)):
                    invalid_values = data_arr[(data_arr < 0) | (data_arr >= self.num)]
                    raise ValueError(
                        f"Array data contains values out of range [0, {self.num - 1}]: {invalid_values} (for {self.num} classes)."
                    )
            
            # Perform one-hot encoding for the 1D array
            one_hot = np.zeros((data_arr.size, self.num), dtype=np.float32)
            if data_arr.size > 0:  # Avoid indexing on an empty array
                one_hot[np.arange(data_arr.size), data_arr] = 1
        
        result['data'] = one_hot
        return result