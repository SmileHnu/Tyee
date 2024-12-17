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
import ast


def get_nested_field(cfg: dict, path: str, default=None):
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

    nesetd_keys = path.split('.')
    lookingup = cfg.copy()
    if not isinstance(lookingup, dict):
        raise ValueError(f"Expected 'cfg' to be a dictionary, but got {type(lookingup)}")

    for nested_key in nesetd_keys:
        # 如果是最后一个键，则直接返回值
        if nested_key == nesetd_keys[-1]:
            lookingup = lookingup.get(nested_key, default)
        else:
            lookingup = lookingup.get(nested_key, {})
    return lookingup


def convert_to_literal(ori_str: str):
    """
    将字符串转换为字面量值，支持转换为布尔值、整数、浮点数、列表或字符串。
    
    :param str ori_str: 需要转换的字符串值。
    :return : 转换后的值（布尔值、整数、浮点数、列表或字符串）。
    """
    ori_str = ori_str.strip()
    if ori_str.lower() == "true":
        return True
    if ori_str.lower() == "false":
        return False

    try:
        return ast.literal_eval(ori_str)
    except (ValueError, SyntaxError):
        return ori_str


def merge_config(cfg: dict, args: dict):
    """
    将命令行参数合并到 YAML 配置中。根据命令行中的键值对更新配置字典，支持多层级配置。
    如果配置中已经存在某个键，则合并其值，否则新增该键。

    :param cfg: dict, 原始配置字典, 包含从 YAML 配置文件中读取的配置。
    :param args: dict, 命令行参数字典, 含通过 `argparse` 获取的参数和配置项。
    :return: dict, 更新后的配置字典, 包含合并后的配置项。
    """
    for k, vs in args.items():
        if k == 'config' or vs is None:
            continue

        if k not in cfg:
            cfg[k] = {}

        for v in vs:
            # 拆分键和值
            if '=' in v:
                vk, vv = v.split('=', 1)
                vv = convert_to_literal(vv)
            else:
                vk, vv = v, True  # 如果没有指定值，默认为 True

            # 处理多层级键
            keys = vk.split('.')
            temp_cfg = cfg[k]
            for key in keys[:-1]:
                temp_cfg = temp_cfg.setdefault(key, {})

            # 设置最终键的值
            temp_cfg[keys[-1]] = vv

    return cfg

def convert_sci_notation(data):
    """
    递归地将配置中的科学计数法字符串转换为浮点数。
    :param data: 配置数据，可以是字典、列表或其他类型。
    :return: 转换后的配置数据。
    """
    if isinstance(data, dict):
        return {k: convert_sci_notation(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_sci_notation(item) for item in data]
    elif isinstance(data, str):
        try:
            # 尝试将字符串转换为浮点数
            return float(data)
        except ValueError:
            return data
    else:
        return data