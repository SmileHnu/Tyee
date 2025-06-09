#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : mit_bih_dataset.py
@Time    : 2025/03/25 18:57:41
@Desc    : 
"""

import os
import re
import mne
import torch
import wfdb
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class MITBIHDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './mit-bih-arrhythmia-database-1.0.0',
        start_offset: float = -64/360,
        end_offset: float = 64/360,
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
    
        
    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        data_files = os.listdir(root_path)
        data_files = np.array(data_files)
        integers = [int(re.search(r'\d+', item).group()) for item in data_files if re.search(r'\d+', item)]
        file_int = set(integers)
        file_int = list(file_int)
        if 256 in file_int:
            file_int.remove(256)
        file_list = [os.path.join(root_path, str(item)) for item in file_int]
        file_list = sorted(file_list)
        print(file_list)
        return file_list

    def read_record(self, record: str, **kwargs):
        
        data = wfdb.rdsamp(record)
        annotation = wfdb.rdann(record,'atr')
        ecg_data = data[0].transpose()
        freq = data[1]['fs']
        ecg_channels = data[1]['sig_name']
        R_location = np.array(annotation.sample)
        labels = [str(symbol) for symbol in annotation.symbol]
        ecg = {
            'data': ecg_data,
            'channels': ecg_channels,
            'freq': freq
        }
        segments = []
        for i in range(len(R_location)):
            segments.append({
                'start': R_location[i] / freq,
                'end': R_location[i]/freq,
                'value':{
                    'symbol':{
                        'data': labels[i],
                    }
                }
            })
        return {
            'signals':{
                'ecg': ecg,
            },
            'labels': {
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
    ) -> Generator[Dict[str, Any], None, None]:
        signals = self.apply_transform(self.before_segment_transform, signals)
        if signals is None:
            print(f"Skip file {meta['file_name']} due to transform error.")
            return None
        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_label = segment['labels']
            seg_info = segment['info']
            symbol = seg_label['symbol']['data']
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
                'symbol': symbol,
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )

    def get_subject_id(self, file_name) -> str:
        return file_name
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        return str(idx)

# from dataset.mit_bih_dataset import MITBIHDataset
# from dataset.transform import PickChannels, Mapping, ZScoreNormalize


# before_segment_transform = [
#     PickChannels(channels=['MLII'], source='ecg', target='ecg'),
#     ZScoreNormalize(axis=-1, source='ecg', target='ecg'),
# ]
# # 1. N - Normal
# # 2. V - PVC (Premature ventricular contraction)
# # 3. / - PAB (Paced beat)
# # 4. R - RBB (Right bundle branch)
# # 5. L - LBB (Left bundle branch)
# # 6. A - APB (Atrial premature beat)
# # 7. ! - AFW (Ventricular flutter wave)
# # 8. E - VEB (Ventricular escape beat)
# offline_label_transform = [
#     Mapping(mapping={
#         'N': 0,
#         'V': 1,
#         '/': 2,
#         'R': 3,
#         'L': 4,
#         'A': 5,
#         '!': 6,
#         'E': 7,
#     }, source='label', target='label'),
# ]

# dataset = MITBIHDataset(
#     root_path='/mnt/ssd/lingyus/test',
#     io_path='/mnt/ssd/lingyus/tyee_mit_bih/train',
#     # io_chunks=224,
#     before_segment_transform=before_segment_transform,
#     offline_label_transform=offline_label_transform,
#     # offline_signal_transform=offline_signal_transform,
#     # online_signal_transform=online_signal_transform,
#     io_mode='hdf5',
#     io_chunks=128,
#     num_worker=4,
# )
# print(len(dataset))
# print(dataset[0]['ecg'].shape)