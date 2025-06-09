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
import numpy as np
from argparse import ArgumentParser
from torch import multiprocessing as mp

from trainer import Trainer
from utils import init_logging
from utils.log_utils import create_experiment_directories, init_logging, save_config
from utils.distributed_utils import call_main
from utils.cfg_utils import get_nested_field, merge_config, convert_sci_notation

# Argument parser for command-line options
parser = ArgumentParser(description="The Physiological signal Representation Learning (PRL) framework.")
parser.add_argument("-c", "--config", type=str, help="Path to the YAML configuration for PRL", required=True)
parser.add_argument("--trainer", nargs='+', help="Override trainer configurations in the format key=value", required=False)
parser.add_argument("--dataset", nargs='+', help="Override dataset configurations in the format key=value", required=False)
parser.add_argument("--model", nargs='+', help="Override model configurations in the format key=value", required=False)
parser.add_argument("--task", nargs='+', help="Override task configurations in the format key=value", required=False)
parser.add_argument("--optimizer", nargs='+', help="Override optimizer configurations in the format key=value", required=False)
parser.add_argument("--lr_scheduler", nargs='+', help="Override learning rate scheduler configurations in the format key=value", required=False)
parser.add_argument("--common", nargs='+', help="Override common configurations in the format key=value", required=False)
parser.add_argument("--distributed", nargs='+', help="Override distributed configurations in the format key=value", required=False)

logger = logging.getLogger(__name__)

def load_cfg() -> dict:
    """
    Load and process the YAML configuration file.

    Returns:
        dict: The processed configuration dictionary.
    """
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Input config file {args.config} does not exist!")

    # Load the YAML configuration file
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Merge command-line arguments into the configuration
    args = dict(vars(args))
    cfg = merge_config(cfg, args)

    # Convert scientific notation in the configuration
    cfg = convert_sci_notation(cfg)

    # Set random seed for reproducibility
    seed = get_nested_field(cfg, "common.seed", 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False    
    return cfg

def main(cfg: dict, rank: int, world_size: int, **kwargs):
    """
    Main function to initialize and run the Trainer.

    Args:
        cfg (dict): The configuration dictionary.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
    """
    trainer = Trainer(cfg, rank, world_size)
    trainer.run()

if __name__ == "__main__":
    # Load configuration
    cfg = load_cfg()

    # Get experiment root directory and task name
    root = get_nested_field(cfg, "common.root", "./experiments/")
    task_select = get_nested_field(cfg, "task.select", "default_task")
    exp_dir = f"{root}/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}-{task_select}"

    # Create experiment directories
    tb_dir, checkpoint_dir = create_experiment_directories(exp_dir)

    # Initialize logging
    init_logging(exp_dir)
    logger = logging.getLogger(__name__)

    # Save the configuration file
    save_config(cfg, exp_dir)

    # Update configuration with experiment paths
    cfg["common"]["exp_dir"] = exp_dir
    cfg["common"]["tb_dir"] = tb_dir
    cfg["common"]["checkpoint_dir"] = checkpoint_dir

    # Call the main function with distributed training support
    call_main(cfg, main)


