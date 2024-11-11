#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : data_utils.py
@Time    : 2024/11/11 19:01:00
@Desc    : 
"""


from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler

def build_dis_sampler(dataset: Dataset, world_size: int, rank: int):
    """
    构建分布式数据加载器的采样器，根据给定的进程数量和进程编号，划分数据集并进行分布式训练。
    
    :param dataset: Dataset, 要进行分布式采样的数据集。
    :param world_size: int, 总进程数，即分布式训练的总工作节点数。
    :param rank: int, 当前进程的编号，用于标识不同的工作节点。
    :return: DistributedSampler, 用于分布式训练的数据采样器。
    :raises TypeError: 如果 dataset 不是 Dataset 类型。
    :raises ValueError: 如果 world_size 或 rank 的值不合法。
    :raises RuntimeError: 如果创建 DistributedSampler 失败。
    """

    # 检查dataset类型
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected 'dataset' to be of type Dataset, but got {type(dataset)}")

    # 检查world_size和rank是否为正整数
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"Expected 'world_size' to be a positive integer, but got {world_size}")
    
    if not isinstance(rank, int) or rank < 0 or rank >= world_size:
        raise ValueError(f"Expected 'rank' to be a valid integer in the range [0, {world_size - 1}], but got {rank}")

    # 创建并返回DistributedSampler
    try:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    except Exception as e:
        raise RuntimeError(f"Failed to create DistributedSampler: {e}")

    return sampler

def build_data_loader(dataset: Dataset, batch_size: int, sampler:Sampler):
    """
    创建数据加载器，用于根据给定的批大小和采样器加载数据。
    
    :param dataset: Dataset, 用于训练或评估的数据集。
    :param batch_size: int, 每个 batch 的数据量。
    :param sampler: Sampler, 用于数据分配的采样器。如果为 None，则默认按照顺序加载数据。
    :return: DataLoader, 用于加载数据的 DataLoader 实例。
    :raises TypeError: 如果 dataset 不是 Dataset 类型，或者 sampler 不是 Sampler 类型。
    :raises ValueError: 如果数据集为空并且没有提供采样器。
    :raises RuntimeError: 如果创建 DataLoader 失败。
    """
     # 检查dataset类型
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected 'dataset' to be of type torch.utils.data.Dataset, but got {type(dataset)}")
    
    # 检查batch_size类型
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"Expected 'batch_size' to be a positive integer, but got {batch_size}")
    
    # 检查sampler类型
    if sampler is not None and not isinstance(sampler, Sampler):
        raise TypeError(f"Expected 'sampler' to be of type torch.utils.data.Sampler, but got {type(sampler)}")
    
    # 如果sampler是None，确保dataset大小不为0
    if sampler is None and len(dataset) == 0:
        raise ValueError("Dataset is empty, cannot create DataLoader without a sampler.")
    
    # 创建DataLoader
    # 如果sampler是None，保持数据原有顺序构造加载器
    if sampler is None:
        try:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataLoader: {e}")

    else:
        try:
            data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=dataset.collate_fn)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataLoader: {e}")
    
    return data_loader