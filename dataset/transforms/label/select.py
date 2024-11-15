#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : select.py
@Time    : 2024/11/13 20:46:30
@Desc    : 
"""
from ..base_transform import BaseTransform


class Select(BaseTransform):
    def __init__(
            self,
            select: list[str]
        ) -> None:
        self.select = select

    def __call__(self, *args, **kwargs):
        
        raise NotImplementedError