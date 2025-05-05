#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : compose.py
@Time    : 2025/03/03 14:30:17
@Desc    : 
"""


from dataset.transform import BaseTransform
from typing import Dict, Any, List, Optional

class Compose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize the Compose transform.

        Args:
            transforms (List[BaseTransform]): List of transforms to compose.
            source (Optional[str]): Not used for sub-transforms. Should be None.
            target (Optional[str]): Not used for sub-transforms. Should be None.

        Raises:
            ValueError: If any sub-transform has source or target set.
        """
        super().__init__(source, target)
        for t in transforms:
            if t.source is not None or t.target is not None:
                raise ValueError("Sub-transforms should not set source or target.")
        self.transforms = transforms

    def transform(self, result: Any) -> Any:
        """
        Sequentially apply all transforms in the list to the input.

        Args:
            result (Any): The input data to be transformed.

        Returns:
            Any: The transformed data after all transforms are applied.
        """
        for t in self.transforms:
            result = t({'data': result})['data'] if t.source or t.target else t(result)
        return result