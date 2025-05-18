#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mapping.py
@Time    : 2025/05/07 10:17:45
@Desc    : 
"""

from typing import Dict, Any
from .base_transform import BaseTransform

class Mapping(BaseTransform):
    def __init__(self, mapping: dict, source: str = None, target: str = None):
        super().__init__(source, target)
        self.mapping = mapping

    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        value = result['data']
        result['data'] = self.mapping[value]
        return result