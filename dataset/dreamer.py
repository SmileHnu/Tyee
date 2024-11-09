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
from pathlib import Path
from torch.nn import functional as F
from base_dataset import BaseDataset



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

    
class DreamerDataset(BaseDataset):
    def __init__(self, source: Path, clip_length: int = 4, split: str = "train") -> None:
        super().__init__()
        random.seed(575)

        if not source.exists():
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        
        self.data, self.targets = self._load_data(source, clip_length=clip_length, split=split)
    
    def _load_data(self, source: str, clip_length: int = 2, split: str = "train"):
        dreamer = Dreamer(source)
        data, target = [], []
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
                    data.append(sample[start:start+sr*clip_length])
                    target.append(arousal[idx])
        # tot = len(data)
        # shuffle_indices = list(range(tot))
        # random.shuffle(shuffle_indices)
        # if split == "train":
        #     data = [data[i] for i in shuffle_indices[:int(tot * 0.8)]]
        #     target = [target[i] for i in shuffle_indices[:int(tot * 0.8)]]
        # else:
        #     data = [data[i] for i in shuffle_indices[int(tot * 0.8):]]
        #     target = [target[i] for i in shuffle_indices[int(tot * 0.8):]]
        return data, target
        
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx] * 1e-3, dtype=torch.float32)
        if self.targets[idx] >= 3:
            target = torch.tensor(1, dtype=torch.long)
        else:
            target = torch.tensor(0, dtype=torch.long)
        with torch.no_grad():
            data = F.layer_norm(data, normalized_shape=data.shape)
        return data, target

    def __len__(self):
        return len(self.targets)

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


if __name__ == "__main__":
    # source = "data\DREAMER.mat"
    source = "/home/taoz/data/PhysioSignal/dreamer/DREAMER.mat"
    dataset = DreamerDataset(Path(source))
    data = dataset[0]
    pass
