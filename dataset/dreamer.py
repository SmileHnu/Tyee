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
import numpy as np
import scipy.io as sio
from pathlib import Path
from base_dataset import BaseDataset
import librosa


def unwrap_squeeze_value(value):
    """
    提取字段的值，如果值的形状为[1]、[1,1]等，则提取标量值。
    """
    if isinstance(value, np.ndarray):
        # 检查是否是单元素数组，形状为 [1], [1,1], [1,1,1], 等
        if value.size == 1:
            # 提取标量值
            # return value.item()
            # 递归提取单个元素的标量值 .item() 提取后的值可能还包含了 [1] 等情况，需要递归处理
            return unwrap_squeeze_value(value.item())
        else:
            # 否则使用 np.squeeze 去除单个维度，保持数据的形状
            return np.squeeze(value)
    return value


def parse_mat_struct(raw_data):
    if isinstance(raw_data, np.ndarray) and raw_data.dtype.names:
        out = {}
        for field in raw_data.dtype.names:
            # 如果字段本身是一个结构体（numpy.void），递归处理
            out[field] = parse_mat_struct(raw_data[field])
        return out
    
    elif isinstance(raw_data, np.ndarray):
        unwraps = unwrap_squeeze_value(raw_data)
        if isinstance(unwraps, np.ndarray) and len(unwraps) > 1:
            for unwrap in unwraps:
                return parse_mat_struct(unwrap)
        # 解析出来了具体的值
        else:
            return unwraps
        # return [parse_single_item(sub_item) for sub_item in item] if item.ndim > 0 else extract_field_value(item)
    elif isinstance(raw_data, np.uint8):
        return raw_data
    else:
        raise Exception(f"Unsupport parsed type: {type(raw_data)}")

    
class DreamerDataset(BaseDataset):
    def __init__(self, source: Path) -> None:
        super().__init__()
        if not source.exists():
            raise FileExistsError(f"Input dataset path {source} not exists !!!")
        self.data,self.len = self._load_data(source)
    
    def _load_data(self, source: str):
        mat = sio.loadmat(source)
        print(mat.keys())
        dreamer = mat["DREAMER"]
        # print(type(dreamer), dreamer.shape)
        # print( type(dreamer[0, 0]) )
        # print( dreamer[0, 0][0] )
        # for data in dreamer[0, 0]:
        #     print(data)
        parsed = {}
        
        parsed = parse_mat_struct(dreamer)
        print(len(parsed))

        print(parsed["Data"].keys())
        
        return parsed["Data"],parsed["noOfSubjects"]
        
        
    
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
        return self.len

    def collate_fn(self, samples):
        wavs, labels = [], []
        for wav, label in samples:
            wavs.append(wav)
            labels.append(label)
        return wavs, labels


if __name__ == "__main__":
    source = "data\DREAMER.mat"
    dataset = DreamerDataset(Path(source))
    data = dataset[0]
    pass
