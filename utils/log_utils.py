#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : log_utils.py
@Time    : 2024/11/19 21:10:35
@Desc    : 
"""
import os
import yaml
import logging

def init_logging(log_dir: str):
    """
    初始化日志记录器，设置基本的日志配置。

    :param log_dir: str, 日志文件存放的目录。
    :return: None, 配置好的日志记录器和文件已经初始化。
    """
    # 创建日志文件夹
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")

    # 设置日志格式和配置
    log_format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    # 只进行一次日志配置
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
    )
    print("配置成功！")

def save_config(cfg, exp_dir):
    """将配置文件保存到experiment目录下的config.yaml。"""
    config_file = os.path.join(exp_dir, "config.yaml")
    with open(config_file, "w", encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

def create_experiment_directories(exp_dir):
    """创建实验目录结构，包括保存log, config, checkpoint, 和 TensorBoard文件夹。"""
    # 创建实验根目录
    os.makedirs(exp_dir, exist_ok=True)

    # 创建TensorBoard文件夹
    tb_dir = os.path.join(exp_dir, f"tb")
    os.makedirs(tb_dir, exist_ok=True)

    # 创建Checkpoint文件夹
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return tb_dir, checkpoint_dir
