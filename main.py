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
from trainer import Trainer


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

def convert_value(value):
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

def merge_config(cfg, args):
    # merge argparser args to yaml config
    for k, vs in args.items():
        if k == 'config':
            continue
        if vs is not None:
            if k not in cfg: cfg[k] = {}
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
    with open(args['config'], 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    trainer = Trainer(cfg)

    trainer_config = cfg.get('trainer',{})
    world_size = trainer_config.get('world_size',1)

    mp.spawn(trainer.train, args=(world_size,), nprocs=world_size, join=True)

    
    
    
    
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


