#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : cfg_utils.py
@Time    : 2024/11/11 19:00:41
@Desc    : 
"""


def get_attr_from_cfg(cfg: dict, path: str, default=None):
    """
    从配置中获取值，支持路径获取
    :param dict cfg: 字典的配置
    :param str path: 提取属性的路径
    :param default: 默认的属性值
    :raises ValueError: 如果cfg不是字典, 则抛出异常
    :return : 返回提取的属性值
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected 'cfg' to be a dictionary, but got {type(cfg)}")

    keys = path.split('.')
    value = cfg

    for key in keys:
        if not isinstance(value, dict):
            print(f"Warning: '{key}' is not a valid key at path '{path}', returning `{default}` for default value.")
            return default
        
        value = value.get(key, default)
        if value == default:
            print(f"Warning: Key '{key}' not found in path '{path}', using `{default}` for default value.")
            return default

    return value