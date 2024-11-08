from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler

def build_dis_sampler(dataset: Dataset, world_size, rank):
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

def build_data_loader(dataset: Dataset, batch_size, sampler:Sampler):

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