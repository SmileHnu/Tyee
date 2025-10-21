#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : wesad_dataset.py
@Time    : 2025/03/30 15:42:48
@Desc    : 
"""
import os
import torch
import scipy
import numpy as np
from pathlib import Path
from typing import Callable, Union, Dict, List, Tuple
from tyee.dataset import BaseDataset

class DaLiADataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './PPG_FieldStudy',
        start_offset: float = 0,
        end_offset: float = 0,
        include_end: bool = False,
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        lazy_threshold: int = 128,
        verbose: bool = True,
    ) -> None:
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'include_end': include_end,
            'before_segment_transform': before_segment_transform,
            'offline_signal_transform': offline_signal_transform,
            'offline_label_transform': offline_label_transform,
            'online_signal_transform': online_signal_transform,
            'online_label_transform': online_label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_chunks': io_chunks,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'lazy_threshold': lazy_threshold,
            'verbose': verbose
        }
        super().__init__(**params)
    
    def set_records(self, root_path, **kwargs):        
        records = list(
            # glob.glob(
            #     os.path.join(*[root_path, "DaLia/PPG_FieldStudy", "S*", "S*.pkl"])
            # )
            Path(root_path).rglob("S*.pkl")
        )
        records = sorted(records)
        return records
    
    def read_record(
        self,
        record: tuple | str,
        **kwargs
    ) -> Dict:
        fname = record

        ppg_freq = 64.0
        acc_freq = 32.0

        ds = ds = np.load(fname, allow_pickle=True, encoding="bytes")
        name = os.path.split(fname)[1][:-4]
        acc_data = ds[b"signal"][b"wrist"][b"ACC"]
        ppg_data = ds[b"signal"][b"wrist"][b"BVP"]
        hr_data = ds[b"label"]
        ppg = {
            "data": ppg_data.T,
            "freq": ppg_freq
        }
        acc = {
            "data": acc_data.T,
            "freq": acc_freq
        }
        hr = {
            "data": hr_data
        }
        segments = []
        segments.append({
            "start": 0,
            "end": len(ppg_data) / ppg_freq,
            "value":{
                'hr': hr
            }
        })
        return {
            'signals':{
                'ppg': ppg,
                'acc': acc
            },
            'labels':{
                'segments': segments,
            },
            'meta':{
                'file_name': os.path.splitext(os.path.basename(record))[0]
            }

        }
        

    def process_record(
        self,
        signals,
        labels,
        meta,
        **kwargs
    ):
        signals = self.apply_transform(self.before_segment_transform, signals)
        if signals is None:
            print(f"Skip file {meta['file_name']} due to transform error.")
            return None
        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_label = segment['labels']
            seg_info = segment['info']
            # print(signals['eeg']['data'].shape)
            # print(label['label']['data'])
            segment_id = self.get_segment_id(meta['file_name'], idx)
            seg_signals = self.apply_transform(self.offline_signal_transform, seg_signals)
            seg_label = self.apply_transform(self.offline_label_transform, seg_label)
            if seg_signals is None or seg_label is None:
                print(f"Skip segment {segment_id} due to transform error.")
                continue
            
            seg_info.update({
                'subject_id': self.get_subject_id(meta['file_name']),
                'session_id': self.get_session_id(),
                'segment_id': self.get_segment_id(meta['file_name'], idx),
                'trial_id': self.get_trial_id(idx),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )
        
    def get_subject_id(self, file_name) -> str:
        return int(file_name[1:])
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    


# from dataset.dalia_dataset import DaLiADataset
# from dataset.transform import WindowExtract,SlideWindow, ForEach, Filter,Detrend,\
#                             ZScoreNormalize, Lambda, Resample, Pad, FFTSpectrum,\
#                             Stack, ExpandDims, Crop, Select, Compose, Mean
# iir_params = dict(order=4, ftype='butter')
# offline_signal_trasnform=[
#     Compose(
#         transforms=[
#             SlideWindow(window_size=8*64, stride=2*64),
#             WindowExtract(),
#             ForEach(
#                 transforms=[
#                     Detrend(),
#                     Filter(l_freq=0.4, h_freq=4, method='iir', phase='forward', iir_params=iir_params),
#                     ZScoreNormalize(axis=-1, epsilon=1e-10),
#                     # Lambda(lambd=lambda x: x.mean(axis=0)),
#                     Mean(axis=0),
#                     Resample(desired_freq=25, window='boxcar', pad='constant', npad=0,),
#                     # Pad(axis=0, side='post', constant_values=0, pad_len=535-200)
#                     FFTSpectrum(resolution=535, min_hz=0.5, max_hz=3.5),
#                 ]),
#         ],source='ppg', target='ppg_spec'
#     ),
#     Compose(
#         transforms=[
#             SlideWindow(window_size=8*32, stride=2*32),
#             WindowExtract(),
#             ForEach(
#                 transforms=[
#                     Detrend(),
#                     Filter(l_freq=0.4, h_freq=4, method='iir', phase='forward', iir_params=iir_params),
#                     ZScoreNormalize(axis=-1, epsilon=1e-10),
#                     Resample(desired_freq=25, window='boxcar', pad='constant', npad=0,),
#                     # Pad(axis=0, side='post', constant_values=0, pad_len=535-200)
#                     FFTSpectrum(resolution=535, min_hz=0.5, max_hz=3.5, axis=-1),
#                     Mean(axis=0),
#                 ]),
#         ], source='acc', target='acc'
#     ),
#     Stack(source=['ppg_spec', 'acc'], target='ppg_acc', axis=-1),
#     Compose(
#         transforms=[
#             ZScoreNormalize(epsilon=1e-10),
#             SlideWindow(window_size=7, stride=1, axis=0),
#         ],source='ppg_acc', target='ppg_acc'
#     ),
#     Compose(
#         transforms=[
#             Detrend(),
#             Filter(l_freq=0.1, h_freq=18, method='iir', phase='forward', iir_params=iir_params),
#             Mean(axis=0),
#             # Resample(desired_freq=25, window='boxcar', pad='constant', npad=0, source='ppg', target='ppg'),
#             ExpandDims(axis=-1),
#             ZScoreNormalize(epsilon=1e-10),
#             SlideWindow(window_size=1280, stride=128, axis=0),
#         ],source='ppg', target='ppg_time'
#     ),
#     Select(key=['ppg_time', 'ppg_acc']),
# ]
# offline_label_trasnform=[
#     Compose(
#         transforms=[
#             Crop(crop_left=6),
#             SlideWindow(window_size=1, stride=1, axis=0),
#         ], source='hr', target='hr'
#     )
# ]

# dataset = DaLiADataset(
#     root_path='/mnt/ssd/lingyus/ppg_dalia/PPG_FieldStudy',
#     io_path='/mnt/ssd/lingyus/tyee_ppgdalia/train',
#     # online_signal_transform=online_signal_trasnform
#     offline_signal_transform=offline_signal_trasnform,
#     offline_label_transform=offline_label_trasnform,
#     # online_label_transform=online_label_trasnform,
#     num_worker=4,
#     io_chunks=320,
# )
# # print(dataset[0])
# print(len(dataset))