#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : select.py
@Time    : 2025/04/21 15:49:03
@Desc    : 
"""
from typing import Union, List, Dict, Any, Optional
from .base_transform import BaseTransform

class Select(BaseTransform):
    def __init__(self, key: Union[str, List[str]], source: Optional[str] = None, target: Optional[str] = None):
        """
        Initialize Select transform.

        Args:
            key (Union[str, List[str]]): Key or list of keys to keep in the result dictionary.
            source (Optional[str]): Not used.
            target (Optional[str]): Not used.
        """
        super().__init__(source, target)
        if isinstance(key, str):
            self.keys = [key]
        else:
            self.keys = key

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select specified keys from the input dictionary.

        Args:
            result (Dict[str, Any]): Input dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing only the selected keys.
        """
        return {k: result[k] for k in self.keys if k in result}
