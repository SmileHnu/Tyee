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
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



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



def merge_config(cfg, args):
    # merge argparser args to yaml config
    # for k, vs in args.items():
    #     if vs is not None:
    #         if k not in args: args[k] = vs
    #         for v in vs:
    #             vk, vv = vkvs if len(vkvs := v.split("=")) == 2 else vkvs[0], None
    #             if vk not in args[k]: raise KeyError(f"Input key {vk} not found in {k} config !")
    #             if vv is None:
    #                 cfg[k][vk] = True
    #             else:
    #                 cfg[k][vk] == vv

    pass


def main() -> None:
    args = parser.parse_args()
    if not os.path.exists(args.config): raise FileExistsError(f"Input config file {args.config} not exists !")
    # windows默认编码是gbk，需指定yaml的编码方式utf-8
    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    args = dict(vars(args))
    merge_config(cfg, args)

    # 访问 YAML 配置
    trainer_config = cfg.get('trainer', {})
    dataset_config = cfg.get('dataset', {})
    model_config = cfg.get('model', {})
    task_config = cfg.get('task', {})
    optimizer_config = cfg.get('optimizer', {})
    lr_scheduler_config = cfg.get('lr_scheduler', {})
    
    # 调用
    fp16 = trainer_config.get('fp16', False)
    world_size = trainer_config.get('world_size', 1)
    ddp_backend = trainer_config.get('ddp_backend', 'default')

    dataset_path = dataset_config.get('path', '')
    num_workers = dataset_config.get('num_workers', 0)
    train_set = dataset_config.get('train', '')
    eval_sets = dataset_config.get('eval', [])

    upstream_model = model_config.get('upstream', {}).get('select', '')
    downstream_model = model_config.get('downstream', {}).get('select', '')
    classes = model_config.get('downstream', {}).get('classes', 0)

    task_type = task_config.get('select', '')
    loss_function = task_config.get('loss', {}).get('select', '')
    loss_weights = task_config.get('loss', {}).get('weight', [])

    optimizer_type = optimizer_config.get('select', '')
    lr_scheduler_type = lr_scheduler_config.get('select', '')

    
    # print(f"Trainer - FP16: {fp16}, World Size: {world_size}, DDP Backend: {ddp_backend}")
    # print(f"Dataset - Path: {dataset_path}, Num Workers: {num_workers}, Train: {train_set}, Eval: {eval_sets}")
    # print(f"Model - Upstream: {upstream_model}, Downstream: {downstream_model}, Classes: {classes}")
    # print(f"Task - Type: {task_type}, Loss Function: {loss_function}, Loss Weights: {loss_weights}")
    # print(f"Optimizer Type: {optimizer_type}, LR Scheduler Type: {lr_scheduler_type}")
    
    
    
    # distributed learning


    # run
    



# # 训练函数
# def train(rank, world_size):
#     # 初始化分布式环境
#     dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
#     # 创建模型并将其移动到对应的设备
#     model = SimpleModel().to(rank)
#     ddp_model = DDP(model, device_ids=[rank])
    
#     # 使用随机数据集
#     dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randn(1000, 10))
    
#     # 使用 DistributedSampler 以确保数据被均匀分布到各个进程
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
#     # 优化器
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
#     criterion = nn.MSELoss()

#     # 模型训练
#     ddp_model.train()
#     for epoch in range(5):
#         sampler.set_epoch(epoch)  # 确保每个 epoch 的数据是不同的
#         for batch_idx, (data, target) in enumerate(dataloader):
#             data, target = data.to(rank), target.to(rank)
#             optimizer.zero_grad()
#             output = ddp_model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % 10 == 0 and rank == 0:
#                 print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

#     # 结束分布式进程
#     dist.destroy_process_group()


# # 主函数
# def main():
#     world_size = 4  # 假设使用 4 个 GPU
#     mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


# def main() -> None:

#     pass


if __name__ == "__main__":
    main()


