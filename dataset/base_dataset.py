#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : dataset.py
@Time    : 2024/09/25 17:02:30
@Desc    : 
"""
from enum import Enum
from typing import Callable
from torch.utils.data import Dataset
from utils import lazy_import_module
from dataset.transforms import Compose, Select


class DatasetType(Enum):
    UNKNOWN = "unknown"
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class BaseDataset(Dataset):
    def __init__(
            self,
            split: DatasetType = DatasetType.UNKNOWN,
            pre_transform: Callable = None,
            post_transform: Callable = None,
            label_transform: Callable = None,
        ) -> None:
        super().__init__()
        self.split = split
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.label_transform = label_transform

    
    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels

    def build_transforms(self, cfg: dict):
        pre_transform_cfg = cfg.get("pre_transforms")
        post_transform_cfg = cfg.get("post_transforms")
        label_transform_cfg = cfg.get("label_transforms")
        pre_transform = None
        if pre_transform_cfg is not None:
            assert isinstance(pre_transform_cfg, list), "pre_transforms config must be a list"
            transforms = self._build_common_transform(pre_transform_cfg)
            pre_transform = Compose(transforms)
        
        post_transform = None
        if post_transform_cfg is not None:
            assert isinstance(post_transform_cfg, list), "post_transforms config must be a list"
            transforms = self._build_common_transform(post_transform_cfg)
            post_transform = Compose(transforms)

        label_transform = None
        if label_transform_cfg is not None:
            assert isinstance(label_transform_cfg, list), "label_transforms config must be a list"
            transforms = self._build_common_transform(label_transform_cfg)
            label_transform = Compose(transforms)
        
        return pre_transform, post_transform, label_transform

    def _build_common_transform(self, transform_cfg: list) -> list:
        transforms = []
        for item in transform_cfg:
            if isinstance(item, str):  # 如果是字符串，直接加载并实例化
                transform_cls = lazy_import_module('dataset.transforms', item)
                transforms.append(transform_cls())
            elif isinstance(item, dict):  # 如果是字典，加载模块并传递参数
                transform_name, params = next(iter(item.items()))
                transform_cls = lazy_import_module('dataset.transforms', transform_name)
                # 如果是Select，需要将ref_channels传递进去
                if transform_cls == Select:
                    transforms.append(transform_cls(params, self.REF_CHANNELS))
                    continue
                # 解析参数中的常量路径
                if isinstance(params, dict):
                    # resolved_params = {k: resolve_constant(v) for k, v in params.items()}
                    transforms.append(transform_cls(**params))
                else:
                    # resolved_params = resolve_constant(params)
                    transforms.append(transform_cls(params))
        return transforms
