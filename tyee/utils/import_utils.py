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
import sys
import os
import logging

log = logging.getLogger(__name__)

def lazy_import_module(module_name: str, class_name: str):
    """
    Dynamically imports a class, trying both as a top-level module (for user_dir)
    and as a submodule of 'tyee' (for installed library usage).
    """
    # Path 1: Try to import as a top-level module.
    # This will succeed if the user has provided a user_dir that contains
    # a 'tasks', 'models', etc. folder.
    # e.g., tries to import 'tasks.my_task'
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        log.info(f"Successfully imported '{class_name}' from user-provided or local path: '{module_name}'.")
        return cls
    except (ImportError, AttributeError):
        # This attempt failed, which is expected if the module is not in user_dir
        # or if we are running as an installed package.
        pass

    # Path 2: Try to import as a submodule of the 'tyee' package.
    # This is the robust way for it to work as an installed library.
    # e.g., tries to import 'tyee.tasks.my_task'
    try:
        absolute_module_name = f"tyee.{module_name}"
        module = importlib.import_module(absolute_module_name)
        cls = getattr(module, class_name)
        log.info(f"Successfully imported '{class_name}' from tyee internal path: '{absolute_module_name}'.")
        return cls
    except (ImportError, AttributeError) as e:
        # If both attempts fail, raise an informative error.
        raise ImportError(
            f"Could not import '{class_name}' from '{module_name}' (user path) "
            f"or '{absolute_module_name}' (internal path). Error: {e}"
        )


def import_user_module(user_dir):
    """
    Adds a user-defined directory to the Python path.

    This allows the framework's lazy-loading mechanism to discover and prioritize
    modules (tasks, models, etc.) from the user's directory without eagerly
    importing them.
    """
    if user_dir is None:
        return

    module_path = os.path.abspath(user_dir)
    if not os.path.exists(module_path):
        print(f"User directory not found: {module_path}, skipping.")
        return

    # Add the user directory to the beginning of sys.path.
    # This is the only action needed. Python's import system will now
    # automatically check this path first when importing modules.
    if module_path not in sys.path:
        sys.path.insert(0, module_path)