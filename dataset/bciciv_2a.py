#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : bciciv_2a.py
@Time    : 2024/11/13 21:46:38
@Desc    : 
"""
import torch
from torch.nn import functional as F
import scipy.io as sio
from pathlib import Path
from typing import Callable, Any
from .base_dataset import BaseDataset, DatasetType
# from base_dataset import BaseDataset, DatasetType


class BCICIV2aDataset(BaseDataset):

    CHANNEL_GDF = {
        'EEG-Fz': 'Fz',
        'EEG-0': 'FC3',
        'EEG-1': 'FC1',
        'EEG-2': 'FCz',
        'EEG-3': 'FC2',
        'EEG-4': 'FC4',
        'EEG-5': 'C5',
        'EEG-C3': 'C3',
        'EEG-6': 'C1',
        'EEG-Cz': 'Cz',
        'EEG-7': 'C2',
        'EEG-C4': 'C4',
        'EEG-8': 'C6',
        'EEG-9': 'CP3',
        'EEG-10': 'CP1',
        'EEG-11': 'CPz',
        'EEG-12': 'CP2',
        'EEG-13': 'CP4',
        'EEG-14': 'P1',
        'EEG-15': 'Pz',
        'EEG-16': 'P2',
        'EEG-Pz': 'POz'
    }

    CHANNELS_MAT = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

    def __init__(
            self,
            root: str,
            split: DatasetType = DatasetType.TRAIN,
            pre_transform: Callable[..., Any] = None,
            post_transform: Callable[..., Any] = None,
            label_transform: Callable[..., Any] = None,
            *args,
            **kwargs
        ) -> None:
        super().__init__(pre_transform, post_transform, label_transform)
        self.root = root
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.label_transform = label_transform
        self.data = self._load_data(root, split)

        self.pad = False
        self.drop = False
        self.random_crop = True
        self.max_sample_size = 1e+12
    
    def _load_data(self, root: str, split: DatasetType) -> list:
        if not Path(root).exists():
            raise Exception(f"Directory {root} not exists !")

        if split == DatasetType.TRAIN:
            fpaths = Path(root).rglob("*T.mat")
        elif split == DatasetType.TEST:
            fpaths = Path(root).rglob("*E.mat")
        else:
            fpaths = Path(root).rglob("*T.mat")

        data = []
        for fpath in fpaths:
            samples = sio.loadmat(str(fpath))["data"]
            for run_id in range(0, samples.size):
                # x, trial, y, fs, classes, artifact, gender, age
                # data shape: [L, C]
                x, trial, y, fs, classes, artifacts, gender, age = samples[0, run_id][0, 0]
                # for all trials
                for trial_idx in range(trial.size):
                    start = trial[trial_idx][0]
                    if trial_idx == trial.size - 1:
                        end = x.shape[0]
                    else:
                        end = trial[trial_idx+1][0]
                    trial_data = x[start:end, :]
                    sample = {
                        "id": f"{fpath.stem}_{run_id}_{trial_idx}",
                        "x": trial_data,
                        "y": y[trial_idx][0],
                        "info": {
                            "fs": int(fs[0][0]),
                            "classes": [x[0] for x in classes[0]],
                            "artifact": artifacts.transpose()[0].tolist(),
                            "gender": str(gender[0]),
                            "age": int(age[0][0])
                        }
                    }

                    if self.pre_transform is not None:
                        sample = self.pre_transform(sample)
                        data.extend(sample)
                    else:
                        data.append(sample)
        return data

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]["x"]).float()
        # x = self.data[idx]["x"]
        y = torch.tensor(self.data[idx]["y"]).long()

        x = x.transpose(1, 0)[9].unsqueeze(0)
        x = x[:, 3*250: 6*250]
        with torch.no_grad():
            x = F.layer_norm(x, normalized_shape=x.shape)
        if self.post_transform:
            x = self.post_transform(x)
        return x, y - 1

    def __len__(self):
        return len(self.data)
    
    def collate_fn(
            self,
            samples: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        wavs, labels = [], []

        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        
        sizes = [w.shape[-1] for w in wavs]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        B, chn = len(wavs), wavs[0].shape[0]

        collated_data = wavs[0].new_zeros(B, chn, target_size)
        padding_mask = (
            torch.BoolTensor(collated_data.shape).fill_(False) if self.pad else None
        )
        for i, data in enumerate(wavs):
            chn, num_frames = data.shape
            diff = num_frames - target_size
            if diff == 0:
                # 长度等于 target_size，直接复制
                collated_data[i] = data
            elif diff < 0:
                # 长度小于 target_size，需要填充
                assert self.pad
                collated_data[i] = torch.cat(
                    [data, data.new_full((chn, -diff), 0.0)], dim=-1
                )
                padding_mask[i, :, diff:] = True
            else:
                # 长度大于 target_size，需要裁剪
                collated_data[i], _ = self.crop_to_max_size(data, target_size)
        return {
            "x": collated_data,
            "label": torch.tensor(labels).long(),
            "padding_mask": padding_mask
        }

    def crop_to_max_size(self, raw: torch.Tensor, target_size: int) -> tuple[torch.Tensor, int]:
        """
        crop the raw physio to the label size if the raw physio size is greater than label size
        :param torch.Tensor raw: the raw bio signal which is needed to crop
        :param int target_size: the crop size
        :return tuple[torch.Tensor, int]: the cropped bio signal and crop start position
        """
        assert raw.dim() == 2, "crop_to_max_size only support 2-dim data"
        _, size = raw.shape
        diff = size - target_size
        # if the data physio size if lower than label size, skip the crop ops
        if diff <= 0:
            return raw, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return raw[:, start:end], start



if __name__ == "__main__":
    from transforms import Clip, Compose, Transpose, ToTensor, Select
    d = BCICIV2aDataset(
        root="/home/taoz/data/PhysioSignal/BCICIV2a",
        split=DatasetType.TRAIN,
        # pre_transform=Clip(start=3, end=6),
        # post_transform=Compose([
        #     ToTensor(),
        #     Transpose([1, 0]),
        #     Select("Cz", BCICIV2aDataset.CHANNELS_MAT),
        # ])
    )
    for x, y in iter(d):
        print(x.shape, y)
        # print(x)
    pass
