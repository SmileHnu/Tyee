#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : tsu_benchmark_dataset.py
@Time    : 2024/11/13 20:17:00
@Desc    : 
"""
import scipy.io as sio
from pathlib import Path
from base_dataset import BaseDataset


class TSUBenchmarkDataset(BaseDataset):
    def __init__(
            self,
            root: str
        ) -> None:
        super().__init__()
        self.root = root
        self._load_data(self.root)

    def _load_data(
            self,
            root: str
        ) -> list:
        if not Path(root).exists():
            raise FileNotFoundError(f"File {root} not exists !")
        
        freq_phase = sio.loadmat(Path(root) / "Freq_Phase.mat")
        freqs = freq_phase["freqs"][0]
        phases = freq_phase["phases"][0]


        samples = sio.loadmat(Path(root) / "S1.mat")
        for trial_idx in range(samples.shape[0]):
            pass
        pass

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


if __name__ == "__main__":
    dataset = TSUBenchmarkDataset(
        root="/home/taoz/data/PhysioSignal/TSUBenckmark"
    )
    pass
