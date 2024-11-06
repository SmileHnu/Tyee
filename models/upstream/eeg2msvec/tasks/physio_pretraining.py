#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : physio_pretraining.py
@Time    : 2024/03/16 18:05:09
@Desc    : 
"""
import os
import sys
import logging
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.dataclass.configs import FairseqDataclass
from omegaconf import MISSING

# from ..data.tuh_eeg_dataset import TuhEEGSegmentDataset

logger = logging.getLogger(__name__)


@dataclass
class PhysioPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sample_rate: float = field(
        default=200,
        metadata={"help": "target sample rate. audio files will be up/down sampled to this rate"},
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    shuffle: Optional[bool] = field(
        default=True,
        metadata={"help": "whether shuffle the wave data file order"},
    )
    pad_to_longest_wave: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    num_per_segment: Optional[int] = field(
        default=12_000,
        metadata={"help": "the segment number to split the physio data"},
    )
    chn_str_lst: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "the channels will be loaded for training, ensure it exist in data file"}
    )


@register_task("physio_pretraining", dataclass=PhysioPretrainingConfig)
class PhysioPretrainingTask(FairseqTask):

    cfg: PhysioPretrainingConfig

    def __init__(
        self,
        cfg: PhysioPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"PhysioPretrainingTask Config {cfg}")

        self.data = cfg.data
        self.sample_rate = cfg.sample_rate
        self.max_sample_size = cfg.max_sample_size
        self.min_sample_size = cfg.min_sample_size
        self.normalize = cfg.normalize
        self.random_crop = cfg.random_crop
        self.shuffle = cfg.shuffle  
        self.pad_to_longest_wave = cfg.pad_to_longest_wave
        self.num_per_segment = cfg.num_per_segment
        self.chn_str_lst = cfg.chn_str_lst

    @classmethod
    def setup_task(
        cls, cfg: PhysioPretrainingConfig, **kwargs
    ) -> "PhysioPretrainingTask":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.data}/{split}.tsv"

        # self.datasets[split] = PhysioSegmentDataset(
        #     manifest_path=manifest,
        #     sample_rate=self.sample_rate,
        #     num_per_segment=self.num_per_segment,
        #     max_sample_size=self.max_sample_size,
        #     min_sample_size=self.min_sample_size,
        #     shuffle=self.shuffle,
        #     pad=self.pad_to_longest_wave,
        #     normalize=self.normalize,
        #     random_crop=self.random_crop,
        # )
        # self.datasets[split] = TuhEEGSegmentDataset(
        #     manifest_path=manifest,
        #     sample_rate=self.sample_rate,
        #     num_per_segment=self.num_per_segment,
        #     max_sample_size=self.max_sample_size,
        #     min_sample_size=self.min_sample_size,
        #     shuffle=self.shuffle,
        #     pad=self.pad_to_longest_wave,
        #     normalize=self.normalize,
        #     random_crop=self.random_crop,
        #     chn_str_lst=self.chn_str_lst
        # )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None
