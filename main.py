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
import logging
import datetime
import utils.log_utils as log_utils
import numpy as np
from trainer import Trainer
from utils import init_logging
from utils.distributed_utils import call_main
from torch import multiprocessing as mp
from utils.cfg_utils import get_nested_field, merge_config, convert_sci_notation


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
parser.add_argument('--distributed', nargs='+', help='The override common configurations in the format key=value', required=False)

logger = logging.getLogger(__name__)

def load_cfg() -> None:
    args = parser.parse_args()
    if not os.path.exists(args.config): 
        raise FileExistsError(f"Input config file {args.config} not exists !")
    
    # windows默认编码是gbk，需指定yaml的编码方式utf-8
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    args = dict(vars(args))
    cfg = merge_config(cfg, args)
    # 转换科学计数法
    cfg = convert_sci_notation(cfg)
    # 加入随机种子
    seed = get_nested_field(cfg, "common.seed", 1337)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    return cfg

def main(cfg, rank, world_size, **kwargs):
    # 实例化 Trainer
    trainer = Trainer(cfg, rank, world_size)
    trainer.train()

if __name__ == "__main__":
    cfg = load_cfg()

    #  获取实验保存路径
    root = get_nested_field(cfg, "common.root", "./experiments/")
    exp_dir = f"{root}/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"
    task_select = get_nested_field(cfg, "task.select", "default_task")

    # 创建实验目录结构
    tb_dir, checkpoint_dir = log_utils.create_experiment_directories(exp_dir, task_select)

    # 配置logging
    log_utils.init_logging(exp_dir)
    logger = logging.getLogger(__name__)

    # 保存配置文件
    log_utils.save_config(cfg, exp_dir)

    # 将路径保存到配置中
    cfg['common']['exp_dir'] = exp_dir
    cfg['common']['tb_dir'] = tb_dir
    cfg['common']['checkpoint_dir'] = checkpoint_dir

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    cuda_visible_devices = get_nested_field(cfg, 'distributed.cuda_visible_devices', None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES to {cuda_visible_devices}")

    call_main(cfg, main)


