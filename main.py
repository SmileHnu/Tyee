#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : main.py
@Time    : 2024/09/23 16:53:18
@Desc    : 
"""
import os
import yaml
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from trainer import Trainer
from utils import get_attr_from_cfg


from argparse import ArgumentParser

parser = ArgumentParser(description="The Physiological signal Representation Learning (PRL) framework.")
parser.add_argument("-c", "--config", type=str, help='Path to the YAML configuration for PRL', required=True)
# parser.add_argument("--trainer", action='append', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--trainer", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--dataset", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--model", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--task", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--optimizer", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--lr_scheduler", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument('--seed', default=1337, type=int)

def convert_value(value: str):
    """
    将字符串值转换为合适的 Python 类型（布尔值、整数、浮点数、列表或字符串）。
    
    尝试将给定的字符串解释为：
    1. 布尔值（'true' 或 'false'）
    2. 整数
    3. 浮点数
    4. 浮点数列表（如果字符串表示一个列表，例如 '[1.0, 2.0, 3.0]'）
    如果以上转换都失败，则返回原始字符串。

    :param value: 需要转换的字符串值。
    :return: 转换后的值（布尔值、整数、浮点数、列表或字符串）。
    """
    # 尝试转换为布尔值
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    # 尝试转换为整数
    try:
        return int(value)
    except ValueError:
        pass
    # 尝试转换为浮点数
    try:
        return float(value)
    except ValueError:
        pass
    # 检查是否为数组列表
    if value.startswith('[') and value.endswith(']'):
        # 去掉方括号并分割，转换为浮点数列表
        return [float(x) for x in value[1:-1].split(',')]
    # 返回原始字符串
    return value

def merge_config(cfg: dict, args: dict):
    """
    将命令行参数合并到 YAML 配置中。根据命令行中的键值对更新配置字典，支持多层级配置。
    如果配置中已经存在某个键，则合并其值，否则新增该键。

    :param cfg: dict，原始配置字典，包含从 YAML 配置文件中读取的配置。
    :param args: dict，命令行参数字典，包含通过 `argparse` 获取的参数和配置项。

    :return: dict，更新后的配置字典，包含合并后的配置项。

    该方法处理以下情况：
        - 如果命令行参数中的 `config` 键存在，则跳过该键。
        - 如果命令行参数中的值为`int` ，则直接赋值
        - 如果命令行参数中的值为 `None`，则在配置中将其值设置为 `True`。
        - 支持多层级的配置更新，例如 `upstream.select` 会将 `select` 配置更新到 `upstream` 下。
        - 对于 `=` 分隔的键值对，进行适当的拆分和赋值，值会根据类型转换。
    """
    for k, vs in args.items():
        if k == 'config':
            continue
        if vs is not None:
            if k not in cfg: cfg[k] = {}
            if isinstance(vs, int):
                cfg[k] = vs
            else:    
                for v in vs:
                    # print(v)
                    vkvs = v.split("=")
                    if len(vkvs) == 2:
                        vk, vv = vkvs
                    else:
                        vk, vv = vkvs[0], None
                    # print(vk,vv)
                    # 处理多层级键upstream.select等
                    keys = vk.split('.')
                    temp_cfg = cfg[k]

                    for key in keys[:-1]:
                        if key not in temp_cfg:
                            raise KeyError(f"Input key '{key}' not found in {k} config!")
                        temp_cfg = temp_cfg[key]
                    
                    # 获取最后一个键进行赋值
                    last_key = keys[-1]
                    
                    if vv is None:
                        temp_cfg[last_key] = True
                    else:
                        temp_cfg[last_key] = convert_value(vv)
    return cfg


def main() -> None:
    args = parser.parse_args()
    print(vars(args))
    if not os.path.exists(args.config): raise FileExistsError(f"Input config file {args.config} not exists !")
    # windows默认编码是gbk，需指定yaml的编码方式utf-8
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    args = dict(vars(args))
    print(cfg)
    cfg = merge_config(cfg, args)

    # 写回
    exp_dir = get_attr_from_cfg(cfg, "common.exp_dir", f"./experiments/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}")
    if not os.path.exists(exp_dir): 
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    # 加入随机种子
    seed = get_attr_from_cfg(cfg,'seed',1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # 训练
    trainer = Trainer(cfg)

    world_size = get_attr_from_cfg(cfg, 'trainer.world_size',1)
    trainer.train(rank=0, world_size=world_size)

    # mp.spawn(trainer.train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()


