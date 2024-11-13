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
from torch import multiprocessing as mp
from trainer import Trainer
from utils.cfg_utils import get_nested_field, merge_config


from argparse import ArgumentParser

parser = ArgumentParser(description="The Physiological signal Representation Learning (PRL) framework.")
parser.add_argument("-c", "--config", type=str, help='Path to the YAML configuration for PRL', required=True)
parser.add_argument("--trainer", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--dataset", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--model", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--task", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--optimizer", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument("--lr_scheduler", nargs='+', help='The override trainer configurations in the format key=value', required=False)
parser.add_argument('--common', nargs='+', help='The override common configurations in the format key=value', required=False)


def main() -> None:
    args = parser.parse_args()
    print(vars(args))
    if not os.path.exists(args.config): 
        raise FileExistsError(f"Input config file {args.config} not exists !")
    
    # windows默认编码是gbk，需指定yaml的编码方式utf-8
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    args = dict(vars(args))
    print(cfg)
    cfg = merge_config(cfg, args)

    # 写回
    root = get_nested_field(cfg, "common.exp_dir", "./experiments/")
    exp_dir = f"{root}/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"

    if not os.path.exists(exp_dir): 
        os.makedirs(exp_dir)
    
    with open(os.path.join(exp_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    # 加入随机种子
    seed = get_nested_field(cfg, "common.seed", 1337)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 训练
    trainer = Trainer(cfg)

    world_size = get_nested_field(cfg, 'trainer.world_size',1)
    trainer.train(rank=0, world_size=world_size)

    # mp.spawn(trainer.train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()


