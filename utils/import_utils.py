#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : import_utils.py
@Time    : 2024/11/11 19:01:26
@Desc    : 
"""
import importlib


def lazy_import_module(module_name: str, class_name: str):
    """
    动态导入指定的模块和类。此函数尝试导入指定模块，并获取该模块中的指定类。
    
    :param module_name: str, 要导入的模块名称。
    :param class_name: str, 要从模块中获取的类名称。
    :return: 类对象, 返回指定模块中的类。
    :raises ImportError: 如果无法导入指定的模块，抛出该异常。
    :raises AttributeError: 如果指定模块中没有该类，抛出该异常。
    """
    try:
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 获取类，如果不存在则抛出 AttributeError
        cls = getattr(module, class_name)
        return cls
    except ImportError as e:
        raise ImportError(f"无法导入模块 '{module_name}': {e}")
    except AttributeError:
        raise AttributeError(f"模块 '{module_name}' 中没有找到类 '{class_name}'")