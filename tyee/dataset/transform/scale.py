#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : scale.py
@Time    : 2025/03/20 11:02:05
@Desc    : 
"""

import numpy as np
from typing import Dict, Any, Optional
from tyee.dataset.transform import BaseTransform

class Scale(BaseTransform):
    def __init__(self, scale_factor: float = 1.0, source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize scaling transform class.

        Args:
        - scale_factor: scaling factor for signal scaling.
        - source: input signal field name.
        - target: output signal field name.
        """
        super().__init__(source, target)
        self.scale_factor = scale_factor

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply numerical scaling to the signal.

        Args:
        - result: dictionary containing signal data with 'data' field.

        Returns:
        - updated signal data dictionary.
        """
        data = result['data']
        result['data'] = data * self.scale_factor
        return result

class Offset(BaseTransform):
    def __init__(self, offset: float | int = 0.0, source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize offset transform class.

        Args:
        - offset: offset value for signal offsetting.
        - source: input signal field name.
        - target: output signal field name.
        """
        super().__init__(source, target)
        self.offset = offset

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply numerical offset to the signal.

        Args:
        - result: dictionary containing signal data with 'data' field.

        Returns:
        - updated signal data dictionary.
        """
        data = result['data']
        result['data'] = data + self.offset
        return result

class Round(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize rounding transform class.

        Args:
        - source: input signal field name.
        - target: output signal field name.
        """
        super().__init__(source, target)

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply rounding to the signal.

        Args:
        - result: dictionary containing signal data with 'data' field.

        Returns:
        - updated signal data dictionary.
        """
        data = result['data']
        result['data'] = np.round(data)
        return result
    
class Log(BaseTransform):
    def __init__(self, epsilon:float=1e-10, source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize logarithmic transform class.

        Args:
        - epsilon: small value to avoid log zero.
        - source: input signal field name.
        - target: output signal field name.
        """
        super().__init__(source, target)
        self.epsilon = epsilon

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply logarithmic transformation to the signal.

        Args:
        - result: dictionary containing signal data with 'data' field.

        Returns:
        - updated signal data dictionary.
        """
        data = result['data']
        result['data'] = np.log(data + self.epsilon)  # 避免对数零点
        return result