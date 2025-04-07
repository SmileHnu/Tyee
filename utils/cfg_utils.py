#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : cfg_utils.py
@Time    : 2024/11/11 19:00:41
@Desc    : Utility functions for configuration management.
"""
import ast


def get_nested_field(cfg: dict, path: str, default=None):
    """
    Retrieve a value from a nested dictionary using a dot-separated path.

    Args:
        cfg (dict): The configuration dictionary.
        path (str): The dot-separated path to the desired value.
        default: The default value to return if the path does not exist.

    Raises:
        ValueError: If `cfg` is not a dictionary.

    Returns:
        Any: The value at the specified path, or the default value if not found.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected 'cfg' to be a dictionary, but got {type(cfg)}")

    nested_keys = path.split('.')
    current = cfg
    for key in nested_keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def convert_to_literal(value: str):
    """
    Convert a string to its literal value (e.g., boolean, int, float, list, or string).

    Args:
        value (str): The string to convert.

    Returns:
        Any: The converted value.
    """
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def merge_config(cfg: dict, args: dict):
    """
    Merge command-line arguments into the YAML configuration.

    Args:
        cfg (dict): The original configuration dictionary (e.g., loaded from a YAML file).
        args (dict): The command-line arguments dictionary.

    Returns:
        dict: The updated configuration dictionary.
    """
    for key, values in args.items():
        if key == 'config' or values is None:
            continue

        if key not in cfg:
            cfg[key] = {}

        for value in values:
            # Split key and value
            if '=' in value:
                nested_key, nested_value = value.split('=', 1)
                nested_value = convert_to_literal(nested_value)
            else:
                nested_key, nested_value = value, True  # Default to True if no value is provided

            # Handle nested keys
            keys = nested_key.split('.')
            temp_cfg = cfg[key]
            for sub_key in keys[:-1]:
                temp_cfg = temp_cfg.setdefault(sub_key, {})

            # Set the final key's value
            temp_cfg[keys[-1]] = nested_value

    return cfg


def convert_sci_notation(data):
    """
    Recursively convert scientific notation strings in a configuration to floats.

    Args:
        data (Any): The configuration data (can be a dict, list, or other types).

    Returns:
        Any: The configuration data with scientific notation strings converted to floats.
    """
    if isinstance(data, dict):
        return {key: convert_sci_notation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_sci_notation(item) for item in data]
    elif isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return data
    else:
        return data