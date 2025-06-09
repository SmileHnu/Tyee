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
    Dynamically import a specified module and class.

    Args:
        module_name (str): The name of the module to import.
        class_name (str): The name of the class to retrieve from the module.

    Returns:
        type: The class object from the specified module.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        # Retrieve the class from the module
        cls = getattr(module, class_name)
        return cls
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_name}': {e}")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")