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
    Initialize the logger with basic configurations.

    Args:
        log_dir (str): Directory where the log file will be stored.

    Returns:
        None
    """
    # Create the log directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")

    # Define the log format
    log_format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Log to the console
            logging.FileHandler(log_file),  # Log to a file
        ]
    )
    print("Logging initialized successfully!")


def save_config(cfg: dict, exp_dir: str):
    """
    Save the configuration dictionary to a YAML file in the experiment directory.

    Args:
        cfg (dict): The configuration dictionary to save.
        exp_dir (str): The experiment directory where the config file will be saved.

    Returns:
        None
    """
    config_file = os.path.join(exp_dir, "config.yaml")
    with open(config_file, "w", encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def create_experiment_directories(exp_dir: str):
    """
    Create the directory structure for an experiment, including subdirectories for logs, checkpoints, and TensorBoard.

    Args:
        exp_dir (str): The root directory for the experiment.

    Returns:
        tuple: Paths to the TensorBoard directory and the checkpoint directory.
    """
    # Create the root experiment directory
    os.makedirs(exp_dir, exist_ok=True)

    # Create the TensorBoard directory
    tb_dir = os.path.join(exp_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # Create the checkpoint directory
    checkpoint_dir = os.path.join(exp_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return tb_dir, checkpoint_dir