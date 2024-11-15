#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : base_transform.py
@Time    : 2024/11/13 20:45:35
@Desc    : 
"""


class BaseTransform(object):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    

