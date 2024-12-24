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

    if not get_nested_field(cfg, 'distributed.distributed_no_spawn', False):
        cfg['distributed']['num_procs'] = min(
            torch.cuda.device_count(), get_nested_field(cfg, 'distributed.world_size', 1)
        )

def _infer_single_node_init(cfg):
    assert (
        get_nested_field(cfg, 'distributed.world_size', 1) <= torch.cuda.device_count()
    ), f"world size is {get_nested_field(cfg, 'distributed.world_size', 1)} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    cfg['distributed']['init_method'] = f"tcp://localhost:{port}"

def distributed_init(cfg):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        warnings.warn(
            "Distributed is already initialized, cannot initialize twice!"
        )
    else:
        dist.init_process_group(
            backend=get_nested_field(cfg, 'distributed.backend', 'nccl'),
            init_method=get_nested_field(cfg, 'distributed.init_method'),
            world_size=get_nested_field(cfg, 'distributed.world_size', 1),
            rank=get_nested_field(cfg, 'distributed.rank'),
        )
        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    cfg['distributed']['rank'] = torch.distributed.get_rank()

    return cfg['distributed']['rank']


def distributed_main(i, main, cfg, kwargs):
    cfg['distributed']['device_id'] = i
    if torch.cuda.is_available() and not get_nested_field(cfg, 'common.cpu', False):
        torch.cuda.set_device(cfg['distributed']['device_id'])
    if get_nested_field(cfg, 'distributed.rank') is None:  # torch.multiprocessing.spawn
        cfg['distributed']['rank'] = kwargs.pop("start_rank", 0) + i

    cfg['distributed']['rank'] = distributed_init(cfg)

    # 重新初始化日志配置
    log_utils.init_logging(cfg['common']['exp_dir'])

    main(cfg, cfg['distributed']['rank'], get_nested_field(cfg, 'distributed.world_size', 1), **kwargs)

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
        if not get_nested_field(cfg, 'distributed.distributed_no_spawn', False):
            start_rank = get_nested_field(cfg, 'distributed.rank')
            cfg['distributed']['rank'] = None  # assign automatically
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
            distributed_main(get_nested_field(cfg, 'distributed.device_id', 0), main, cfg, kwargs)
    else:
        logger.info("Final configuration:\n" + yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        # single GPU main
        main(cfg, 0, 1, **kwargs)