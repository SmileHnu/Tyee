#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : distributed_utils.py
@Time    : 2024/12/23 14:36:48
@Desc    : Utility functions for distributed training in PyTorch.
"""

import torch
import random
import yaml
import logging
import torch.distributed as dist
from tyee.utils.cfg_utils import get_nested_field
from tyee.utils import log_utils


def infer_init_method(cfg, force_distributed=False):
    """
    Infer the initialization method for distributed training.

    Args:
        cfg (dict): The configuration dictionary.
        force_distributed (bool): Whether to force distributed training.

    Returns:
        None
    """
    if get_nested_field(cfg, 'distributed.init_method') is not None:
        return

    if get_nested_field(cfg, 'distributed.world_size', 1) > 1 or force_distributed:
        _infer_single_node_init(cfg)


def _infer_single_node_init(cfg):
    """
    Infer the initialization method for a single-node distributed setup.

    Args:
        cfg (dict): The configuration dictionary.

    Raises:
        AssertionError: If the world size exceeds the number of available GPUs.

    Returns:
        None
    """
    world_size = get_nested_field(cfg, 'distributed.world_size', 1)
    available_devices = torch.cuda.device_count()
    assert world_size <= available_devices, (
        f"World size is {world_size}, but only {available_devices} devices are available."
    )
    port = random.randint(10000, 20000)
    cfg['distributed']['init_method'] = f"tcp://localhost:{port}"


def distributed_init(cfg, rank):
    """
    Initialize the distributed process group.

    Args:
        cfg (dict): The configuration dictionary.
        rank (int): The rank of the current process.

    Raises:
        Warning: If the distributed process group is already initialized.

    Returns:
        None
    """
    if dist.is_available() and dist.is_initialized():
        logging.warning("Distributed is already initialized, cannot initialize twice!")
    else:
        dist.init_process_group(
            backend=get_nested_field(cfg, 'distributed.backend', 'nccl'),
            init_method=get_nested_field(cfg, 'distributed.init_method'),
            world_size=get_nested_field(cfg, 'distributed.world_size', 1),
            rank=rank,
        )
        # Perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())


def distributed_main(rank, main, cfg, kwargs):
    """
    Main function for distributed training.

    Args:
        rank (int): The rank of the current process.
        main (callable): The main function to execute.
        cfg (dict): The configuration dictionary.
        kwargs (dict): Additional arguments for the main function.

    Returns:
        None
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    distributed_init(cfg, rank)

    # Reinitialize logging for each process
    log_utils.init_logging(cfg['common']['exp_dir'])

    main(cfg, rank, get_nested_field(cfg, 'distributed.world_size', 1), **kwargs)

    if dist.is_initialized():
        dist.barrier()

    dist.destroy_process_group()


def call_main(cfg, main, **kwargs):
    """
    Entry point for distributed or single-GPU training.

    Args:
        cfg (dict): The configuration dictionary.
        main (callable): The main function to execute.
        kwargs (dict): Additional arguments for the main function.

    Returns:
        None
    """
    log_utils.init_logging(get_nested_field(cfg, 'common.exp_dir'))
    logger = logging.getLogger(__name__)

    if get_nested_field(cfg, 'distributed.init_method') is None:
        infer_init_method(cfg)

    if get_nested_field(cfg, 'distributed.init_method') is not None:
        # Distributed training
        start_rank = 0
        kwargs["start_rank"] = start_rank
        logger.info("Final configuration:\n" + yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(main, cfg, kwargs),
            nprocs=min(
                torch.cuda.device_count(),
                get_nested_field(cfg, 'distributed.world_size', 1),
            ),
            join=True,
        )
    else:
        # Single-GPU training
        logger.info("Final configuration:\n" + yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        main(cfg, 0, 1, **kwargs)