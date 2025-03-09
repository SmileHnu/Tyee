#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : dreamer.py
@Time    : 2024/09/25 16:58:38
@Desc    : 
"""
import os
import torch
import random
import scipy.io as sio
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from dataset.base_dataset import BaseDataset



class Dreamer(object):

    KEYS = [
        "Data",
        "EEG_SamplingRate",
        "ECG_SamplingRate",
        "EEG_Electrodes",
        "noOfSubjects",
        "noOfVideoSequences",
        "Disclaimer",
        "Provider",
        "Version",
        "Acknowledgement"
    ]

    EMOTION_VIDEO_SEQENCE = [
        "calmness", "surprise", "amusement", "fear", "excitement", "disgust",
        "happiness", "anger", "sadness", "disgust", "calmness", "amusement",
        "happiness", "anger", "fear", "excitement", "sadness", "surprise"
    ]


    def __init__(self, source: str) -> None:
        self.source = source

        self.data = []
        self.eeg_sampling_rate = 128
        self.ecg_sampling_rate = 256
        self.eeg_electrodes = []
        self.no_of_subjects = 0
        self.no_of_video_sequence = 0
        self.disclaimer = ""
        self.provider = ""
        self.version = ""
        self.acknowledgement = ""
        if not os.path.exists(source):
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        self.__raw = sio.loadmat(source)["DREAMER"][0, 0]
        self._parse_additional_info(self.__raw)
        self._parse_dreamer_data(self.__raw)
        pass

    def _parse_additional_info(self, raw) -> None:
        self.eeg_sampling_rate = int(raw["EEG_SamplingRate"][0, 0])
        self.ecg_sampling_rate = int(raw["ECG_SamplingRate"][0, 0])
        for electrode in list(raw["EEG_Electrodes"][0]):
            self.eeg_electrodes.append(str(electrode[0]))
        self.no_of_subjects = int(raw["noOfSubjects"][0, 0])
        self.no_of_video_sequence = int(raw["noOfVideoSequences"][0, 0])
        self.disclaimer = str(raw["Disclaimer"][0])
        self.provider = str(raw["Provider"][0])
        self.version = str(raw["Version"][0])
        self.acknowledgement = str(raw["Acknowledgement"][0])

    def _parse_dreamer_data(self, raw) -> None:

        def extract_baseline_stimuli(sample, signal_type: str = "EEG"):
            signal_data = sample[signal_type][0, 0]
            baseline = [bl[0] for bl in signal_data["baseline"][0, 0].tolist()]
            stimuli = [sl[0] for sl in signal_data["stimuli"][0, 0].tolist()]
            return baseline, stimuli

        def unwrap(data: list):
            return [d[0] for d in data]

        data = raw["Data"][0]
        # for loop to all subject
        for sample in list(data):
            eeg_bl_lst, eeg_sl_lst = extract_baseline_stimuli(sample, "EEG")
            ecg_bl_lst, ecg_sl_lst = extract_baseline_stimuli(sample, "ECG")
            res = {
                "Age": str(sample["Age"][0, 0][0]),
                "Gender": str(sample["Gender"][0, 0][0]),
                "EEG": {
                    "baseline": eeg_bl_lst,
                    "stimuli": eeg_sl_lst
                },
                "ECG": {
                    "baseline": ecg_bl_lst,
                    "stimuli": ecg_sl_lst
                },
                "ScoreValence": unwrap(sample["ScoreValence"][0, 0].tolist()),
                "ScoreArousal": unwrap(sample["ScoreArousal"][0, 0].tolist()),
                "ScoreDominance": unwrap(sample["ScoreDominance"][0, 0].tolist()),
            }
            self.data.append(res)
        pass

    
class DREAMERDataset(BaseDataset):
    def __init__(
            self,
            source: Path,
            clip_length: int = 2,
            split: str = "train",
            pad: bool = False,
            drop: bool = True
        ) -> None:
        super().__init__()
        random.seed(575)
        self.pad = pad
        self.drop = drop
        self.max_sample_size = 1e+12

        if not source.exists():
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        
        self.data, self.targets = self._load_data(source, clip_length=clip_length, split=split)
    
    def _load_data(self, source: str, clip_length: int = 2, split: str = "train"):
        dreamer = Dreamer(source)
        data, label = [], []
        num = int(len(dreamer.data) * 0.8)
        if split == "train":
            curr_data = dreamer.data[:num]
        else:
            curr_data = dreamer.data[num:]
        for subject in curr_data:
            sr = dreamer.eeg_sampling_rate
            subject_data = subject["EEG"]["stimuli"]
            arousal = subject["ScoreArousal"]
            for idx, sample in enumerate(subject_data):
                num_frames = int(sample.shape[0])
                for start in range(0, num_frames, clip_length * sr):
                    if self.drop and start + sr * clip_length > num_frames:
                        continue
                    data.append(sample[start:start+sr*clip_length])
                    label.append(arousal[idx])
        # tot = len(data)
        # shuffle_indices = list(range(tot))
        # random.shuffle(shuffle_indices)
        # if split == "train":
        #     data = [data[i] for i in shuffle_indices[:int(tot * 0.8)]]
        #     label = [label[i] for i in shuffle_indices[:int(tot * 0.8)]]
        # else:
        #     data = [data[i] for i in shuffle_indices[int(tot * 0.8):]]
        #     label = [label[i] for i in shuffle_indices[int(tot * 0.8):]]
        return data, label
        
    def __getitem__(self, idx):
        # shape: [L, C]
        data = torch.tensor(self.data[idx] * 1e-3, dtype=torch.float32)
        # shape: [C, L]
        data = data.transpose(0, 1)

        # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        data = data[2].unsqueeze(0)
        if self.targets[idx] >= 3:
            label = torch.tensor(1, dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)
        with torch.no_grad():
            data = F.layer_norm(data, normalized_shape=data.shape)
        return data, label

    def __len__(self):
        return len(self.targets)

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
    # source = "data\DREAMER.mat"
    source = "/home/taoz/data/PhysioSignal/dreamer/DREAMER.mat"
    dataset = DREAMERDataset(Path(source))
    data = dataset[0]
    pass
