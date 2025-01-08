#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : distributed_utils.py
@Time    : 2024/12/23 14:36:48
@Desc    : 
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import yaml
import socket
import logging
from utils.cfg_utils import get_nested_field
from utils import log_utils



def infer_init_method(cfg, force_distributed=False):
    if get_nested_field(cfg, 'distributed.init_method') is not None:
        return

    if get_nested_field(cfg, 'distributed.world_size', 1) > 1 or force_distributed:
        _infer_single_node_init(cfg)


def _infer_single_node_init(cfg):
    assert (
        get_nested_field(cfg, 'distributed.world_size', 1) <= torch.cuda.device_count()
    ), f"world size is {get_nested_field(cfg, 'distributed.world_size', 1)} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    cfg['distributed']['init_method'] = f"tcp://localhost:{port}"

def distributed_init(cfg, i):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        warnings.warn(
            "Distributed is already initialized, cannot initialize twice!"
        )
    else:
        dist.init_process_group(
            backend=get_nested_field(cfg, 'distributed.backend', 'nccl'),
            init_method=get_nested_field(cfg, 'distributed.init_method'),
            world_size=get_nested_field(cfg, 'distributed.world_size', 1),
            rank=i,
        )
        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    


def distributed_main(i, main, cfg, kwargs):
    
    if torch.cuda.is_available():
        torch.cuda.set_device(i)
    distributed_init(cfg, i)

    # 重新初始化日志配置
    log_utils.init_logging(cfg['common']['exp_dir'])

    main(cfg, i, get_nested_field(cfg, 'distributed.world_size', 1), **kwargs)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dist.destroy_process_group()

def call_main(cfg, main, **kwargs):

    log_utils.init_logging(get_nested_field(cfg, 'common.exp_dir'))
    logger = logging.getLogger(__name__)

    if get_nested_field(cfg, 'distributed.init_method') is None:
        infer_init_method(cfg)

    if get_nested_field(cfg, 'distributed.init_method') is not None:
        # distributed training
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
        logger.info("Final configuration:\n" + yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        # single GPU main
        main(cfg, 0, 1, **kwargs)