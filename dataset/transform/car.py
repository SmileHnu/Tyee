#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : car.py
@Time    : 2025/03/30 20:35:06
@Desc    : 
"""


from typing import Dict, Any, Optional
import numpy as np
from .base_transform import BaseTransform

class CAR(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize the CAR class for Common Average Referencing (CAR).
        """
        super().__init__(source=source, target=target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Common Average Referencing (CAR) to the input signal.

        Args:
            result (Dict[str, Any]): A dictionary containing the signal data, must include the 'data' key.

        Returns:
            Dict[str, Any]: The dictionary with the CAR-transformed signal.
        """
        # Get signal data
        signals = result.get("data")
        if signals is None:
            raise ValueError("The input dictionary must contain the 'data' key.")

        # Check if the signal is 2D or 3D
        if signals.ndim == 2:
            # 2D signal (n_channels, n_times)
            common_average = np.mean(signals, axis=0, keepdims=True)  # Average across channels for each time point
            signals = signals - common_average  # Subtract common average from each channel
        elif signals.ndim == 3:
            # 3D signal (n_epochs, n_channels, n_times)
            common_average = np.mean(signals, axis=1, keepdims=True)  # Average across channels for each epoch
            signals = signals - common_average  # Subtract common average from each channel
        else:
            raise ValueError(f"Unsupported signal dimension: {signals.ndim}D. Only 2D or 3D signals are supported.")

        # Update the result dictionary
        result["data"] = signals
        return result