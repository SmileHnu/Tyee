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
import numpy as np
import scipy.io as sio
from pathlib import Path
from base_dataset import BaseDataset
import librosa


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
                    "stomuli": eeg_sl_lst
                },
                "ECG": {
                    "baseline": ecg_bl_lst,
                    "stomuli": ecg_sl_lst
                },
                "ScoreValence": sample["ScoreValence"][0, 0].tolist(),
                "ScoreArousal": sample["ScoreArousal"][0, 0].tolist(),
                "ScoreDominance": sample["ScoreDominance"][0, 0].tolist(),
            }
            self.data.append(res)
        pass

    
class DreamerDataset(BaseDataset):
    def __init__(self, source: Path) -> None:
        super().__init__()
        if not source.exists():
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        
        self.data, self.num_samples = self._load_data(source)
    
    def _load_data(self, source: str):
        dreamer = Dreamer(source)
        return None, 0
        
    def __getitem__(self, idx):
        
        # eeg_data = self.data[idx]["EEG"]
        # ecg_data = self.data[idx]["ECG"]
        # valence_scores = self.data[idx]["ScoreValence"]
        # arousal_scores = self.data[idx]["ScoreArousal"]
        # dominance_scores = self.data[idx]["ScoreDominance"]

        eeg_data = self.data["EEG"]
        ecg_data = self.data["ECG"]
        valence_scores = self.data["ScoreValence"]
        arousal_scores = self.data["ScoreArousal"]
        dominance_scores = self.data["ScoreDominance"]
        
        print(type(eeg_data))
        print(type(eeg_data))
        print(type(valence_scores))
        print(type(arousal_scores))
        print(type(dominance_scores))

        # MFCC
        eeg_mfcc = [librosa.feature.mfcc(y,sr=128,n_mfcc=13) for y in eeg_data]
        ecg_mfcc = [librosa.feature.mfcc(y,sr=256,n_mfcc=13) for y in ecg_data]

        return{
            eeg_mfcc,
            ecg_mfcc,
            valence_scores,
            arousal_scores,
            dominance_scores
        }

    def __len__(self):
        return self.num_samples

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
