# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys
import logging
from omegaconf import MISSING, II
from typing import Optional, TypeVar
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.tasks import FairseqTask, register_task

try:
    from ..data.tueg_dataset import EEGTUEGDataset
except ImportError:
    pass

logger = logging.getLogger(__name__)



@dataclass
class EEGPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    chns: list[str] = field(
        default_factory=list,
        metadata={"help": "channels to load from the EEG h5 file"},
    )
    sample_rate: int = field(
        default=512,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    tpu: bool = II("common.tpu")
    seed: int = II("common.seed")



@register_task(name="eeg_pretraining", dataclass=EEGPretrainingConfig)
class EEGPretrainingTask(FairseqTask):
    cfg: EEGPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: EEGPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg


        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = EEGTUEGDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            chns=self.cfg.chns,
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            shuffle=True,
            pad=(task_cfg.labels is not None) or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            random_crop=False,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(model_cfg, from_checkpoint)
        return model

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None