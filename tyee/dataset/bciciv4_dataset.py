#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv4_dataset.py
@Time    : 2025/03/28 19:23:58
@Desc    : 
"""

import os
import scipy.io as scio
import numpy as np
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator, List


class BCICIV4Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV4',
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        include_end: bool = False,
        before_segment_transform: Union[None, List[Callable]] = None,
        offline_signal_transform: Union[None, List[Callable]] = None,
        offline_label_transform: Union[None, List[Callable]] = None,
        online_signal_transform: Union[None, List[Callable]] = None,
        online_label_transform: Union[None, List[Callable]] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        verbose: bool = True,
    ) -> None:
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')

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
            'verbose': verbose
        }
        # save all arguments to __dict__
        self.__dict__.update(params)
        super().__init__(**params) 
        
    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('comp.mat'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        
        return file_list

    def read_record(self, record: str, **kwargs):
        
        data = scio.loadmat(record)
        test_label = scio.loadmat(record.replace('comp.mat','testlabels.mat'))
        train_data = data['train_data'].astype(np.float64).T
        train_dg = {
            'data': data['train_dg'].astype(np.float64).T,
            'freq': 1000
        }
        test_data = data['test_data'].astype(np.float64).T
        test_dg = {
            'data': test_label['test_dg'].astype(np.float64).T,
            'freq': 1000
        }
        data = [train_data, test_data]
        data = np.concatenate(data, axis=1)
        ecog = {
            'data': data,
            'channels': [f"{i}" for i in range(data.shape[0])],
            'freq': 1000
        }
        segments = [
            {
                'start': 0,
                'end': train_data.shape[1] / 1000,
                'value':{
                    'dg': train_dg
                }
            },
            {
                'start': train_data.shape[1] / 1000,
                'end': data.shape[1] / 1000,
                'value':{
                    'dg': test_dg
                }
            }
        ]
        
        return {
            'signals':{
                'ecog': ecog,
            },
            'labels':{
                'segments': segments
            },
            'meta':{
                'file_name': os.path.splitext(os.path.basename(record))[0],
            }
        }
  
    def process_record(
        self,
        signals,
        labels,
        meta,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        signals = self.apply_transform(self.before_segment_transform, signals)
        if signals is None:
            print(f"Skip file {meta['file_name']} due to transform error.")
            return None

        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_labels = segment['labels']
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
                labels=seg_labels,
                info=seg_info,
            )

    
    def get_subject_id(self, file_name) -> str:
        return file_name.split('_')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        return [f"{i}_{segment_id}" for i in range(sample_len)]



# test /home/lingyus/data/BCICIV4/sub1_comp.mat
# from dataset.bciciv4_dataset import BCICIV4Dataset
# from dataset.transform import Compose, NotchFilter, Filter, ZScoreNormalize, RobustNormalize, Reshape,\
#                               CWTSpectrum, Downsample, Crop, Interpolate, MinMaxNormalize, CommonAverageRef,\
#                               Transpose
# from dataset.transform.slide_window import SlideWindow


# minmax_stats = np.load('/home/lingyus/data/BCICIV4/sub1/minmax_scaler_stats0.npz')
# data_min_ = minmax_stats['data_min_']
# data_max_ = minmax_stats['data_max_']

# robust_stats = np.load('/home/lingyus/data/BCICIV4/sub1/robust_scaler_stats0.npz')
# center_ = robust_stats['center_']
# scale_ = robust_stats['scale_']
# print(center_.shape, scale_.shape)
# offline_signal_transform = [
    
#     ZScoreNormalize(epsilon=0, axis=1, source='ecog', target='ecog'),
#     CommonAverageRef(axis=0, source='ecog', target='ecog'),
#     Filter(l_freq=40, h_freq=300, source='ecog', target='ecog'),
#     NotchFilter(freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450], source='ecog', target='ecog'),
#     CWTSpectrum(freqs=np.logspace(np.log10(40), np.log10(300), 40), output_type='power', n_jobs=6, source='ecog', target='ecog'),
#     Downsample(desired_freq=100, source='ecog', target='ecog'),
#     Crop(crop_right=20, source='ecog', target='ecog'),
#     Transpose(source='ecog', target='ecog'),
#     Reshape(shape=(-1, 40*62), source='ecog', target='ecog'),
#     RobustNormalize(
#         median=center_, iqr=scale_, 
#         unit_variance=False, quantile_range=(0.1, 0.9), epsilon=0, axis=0, source='ecog', target='ecog'),
#     Reshape(shape=(-1, 40, 62), source='ecog', target='ecog'),
#     Transpose(source='ecog', target='ecog'),
#     SlideWindow(window_size=256, stride=1, source='ecog', target='ecog'),
# ]
# offline_label_transform = [
#         Downsample(desired_freq=25, source='dg', target='dg'),
#         Interpolate(desired_freq=100, kind='cubic', source='dg', target='dg'),
#         Crop(crop_left=20, source='dg', target='dg'),
#         Transpose(source='dg', target='dg'),
#         MinMaxNormalize(
#             min=data_min_, max=data_max_, 
#             axis=0, source='dg', target='dg'),
#         Transpose(source='dg', target='dg'),
#         SlideWindow(window_size=256, stride=1, source='dg', target='dg'),
# ]
# dataset = BCICIV4Dataset(root_path='/home/lingyus/data/BCICIV4/sub1',
#                         io_path='/home/lingyus/data/BCICIV4/sub1/processed_test',
#                         offline_signal_transform=offline_signal_transform,
#                         offline_label_transform=offline_label_transform,
#                         io_mode='hdf5',
#                         io_chunks=256,
#                         num_worker=8)