#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : base_transform.py
@Time    : 2025/02/23 20:44:09
@Desc    : 
"""

from typing import Dict, Any, List, Optional

class BaseTransform:
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            source (Optional[str]): The key of the source signal to transform.
            target (Optional[str]): The key to store the transformed signal.
        """
        self.source = source
        self.target = target

    def __call__(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transform to the specified source in signals and store the result in target.

        Args:
            signals (Dict[str, Any]): The input signals dictionary.

        Returns:
            Dict[str, Any]: The updated signals dictionary if target is set, otherwise the transformed result.
        """
        # Read source data
        signal = signals[self.source] if self.source else signals
        # Apply transform
        out = self.transform(signal)
        # Write to target
        if self.target:
            signals[self.target] = out
            return signals
        return out

    def transform(self, result: Any) -> Any:
        """
        The actual transform logic should be implemented in subclasses.

        Args:
            result (Any): The input data to transform.

        Returns:
            Any: The transformed data.
        """
        raise NotImplementedError("Subclasses must implement this method")