#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : osausd_dataset.py
@Time    : 2025/05/24 13:24:59
@Desc    : 
"""

import os
import torch
import scipy
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Callable, Union, Dict, List, Tuple
from dataset.base_dataset import BaseDataset

class OSAUSDDataset(BaseDataset):
    def __init__(
        self,
        # OSAUSD specific parameters
        nan_replacement_value: float = -1.0,
        discard_threshold: float = 0.5,
        # BaseDataset parameters
        root_path: str = './dataset_OSAS.pickle',
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
        self.nan_replacement_value = nan_replacement_value
        self.discard_threshold = discard_threshold
        super().__init__(**params)
    
    def set_records(self, root_path, **kwargs):        
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        records = [root_path]
        return records
    
    def read_record(
        self,
        record: tuple | str,
        **kwargs
    ) -> Dict:
        dataset = pd.read_pickle(record)
        critical_columns = ['signal_ecg_ii', 'SpO2(%)']  # 指定关键列
        dataset = dataset.dropna(subset=critical_columns)
        patient_map_features = {}
        # 1. 预处理数据，将每个病人的数据整理到 patient_map_features
        for pat in dataset['patient'].unique():
            temp = dataset[dataset['patient'] == pat]
            feature_map_ts = {}
            for col in dataset.columns[1:]:
                if 'signal' not in col and 'PSG_' not in col:
                    feature_map_ts[col] = temp[col].values
                else:
                    feature_map_ts[col] = np.concatenate(temp[col].values)
            patient_map_features[pat] = feature_map_ts

        # 初始化用于存储所有病人拼接后数据的列表
        all_ecg_data = []
        all_ppg_data = []
        all_hr_data = []
        all_spo2_data = []
        all_pi_data = []
        all_rr_data = [] 
        all_pvcs_data = []
        # 存储每个病人在拼接后数据中的 (start_index, end_index)
        segments = []
        current_length = 0

        # 按照病人ID的自然顺序（如果需要排序的话）或原始顺序处理
        # 如果病人ID是数字字符串，可以排序以确保一致性
        sorted_patient_ids = sorted(patient_map_features.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for pat_id_str in sorted_patient_ids:
            data = patient_map_features[pat_id_str]
            
            # 2. 处理和拼接ECG数据
            ecg_i = data['signal_ecg_i']
            ecg_ii = data['signal_ecg_ii']
            ecg_iii = data['signal_ecg_iii']
            
            pat_ecg_data = np.vstack([ecg_i, ecg_ii, ecg_iii])

            # 3. 获取其他信号
            pat_ppg_data = data['signal_pleth']
            pat_hr_data = data['HR(bpm)']
            pat_spo2_data = data['SpO2(%)']
            pat_pi_data = data['PI(%)']
            pat_rr_data = data['RR(rpm)']
            pat_pvcs_data = data['PVCs(/min)']
            # 标签数据
            pat_event_data = data['event']
            pat_anomaly_data = data['anomaly']
            current_pat_length = len(pat_anomaly_data)
            
            # 4. 记录病人的起止时间索引
            start_index = current_length * 1
            end_index = (current_length + current_pat_length) * 1
            segments.append({
                'start': start_index,
                'end':end_index,
                'value': {
                    'event': {
                        'data': pat_event_data
                    },
                    'anomaly': {
                        'data': pat_anomaly_data
                    }
                }
            })
            current_length = end_index
            
            # 5. 将当前病人的数据追加到总列表
            all_ecg_data.append(pat_ecg_data)
            all_ppg_data.append(pat_ppg_data)
            all_hr_data.append(pat_hr_data)
            all_spo2_data.append(pat_spo2_data)
            all_pi_data.append(pat_pi_data)
            all_rr_data.append(pat_rr_data)
            all_pvcs_data.append(pat_pvcs_data)

        # 6. 将所有病人的数据沿时间轴拼接起来
        final_ecg_data = np.concatenate(all_ecg_data, axis=1)
        final_ppg_data = np.concatenate(all_ppg_data)
        final_hr_data = np.concatenate(all_hr_data)
        final_spo2_data = np.concatenate(all_spo2_data)
        final_pi_data = np.concatenate(all_pi_data)
        final_rr_data = np.concatenate(all_rr_data)
        final_pvcs_data = np.concatenate(all_pvcs_data)

        # 7. 准备返回的字典
        derived_freq = 1 
        waveform_freq = 80 

        return {
            'signals': {
                'ecg': {'data': final_ecg_data, 'freq': waveform_freq, 'channels': ['I', 'II', 'III']},
                'ppg': {'data': final_ppg_data, 'freq': waveform_freq},
                'hr': {'data': final_hr_data, 'freq': derived_freq},
                'spo2': {'data': final_spo2_data, 'freq': derived_freq},
                'pi': {'data': final_pi_data, 'freq': derived_freq},
                'rr': {'data': final_rr_data, 'freq': derived_freq}, 
                'pvcs': {'data': final_pvcs_data, 'freq': derived_freq},
            },
            'labels': {
                'segments': segments,
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(record))[0],
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
            seg_signals, seg_label = self._handle_segment_missing_values_by_windows(
                seg_signals=seg_signals,
                seg_label=seg_label,
                segment_log_id=segment_id,
                nan_replacement_value=self.nan_replacement_value,
                discard_threshold=self.discard_threshold
            )
            seg_info.update({
                'subject_id': self.get_subject_id(idx),
                'session_id': self.get_session_id(),
                'segment_id': self.get_segment_id(meta['file_name'], idx),
                'trial_id': self.get_trial_id(),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )
        
    def get_subject_id(self, idx) -> str:
        # Extract the subject ID from the file name
        # Assuming the file name format is like "subject_id_record_id"
        # You can modify this logic based on your actual file naming convention
        return idx
    
    def get_segment_id(self, file_name, idx) -> str:
        # Extract the segment ID from the file name
        # Assuming the segment ID is the same as the file name in this case
        return f'{idx}_{file_name}'
    
    def _handle_segment_missing_values_by_windows(
        self, 
        seg_signals: Dict, 
        seg_label: Dict,
        segment_log_id: str, 
        nan_replacement_value: float = -1.0,
        discard_threshold: float = 0.5
    ) -> Union[Tuple[Dict, Dict], None]:
        """
        基于每个信号的windows信息处理缺失值：
        1. 检查每个窗口内的缺失值比例
        2. 记录需要删除的窗口索引
        3. 统一删除所有信号和标签对应的窗口索引
        4. 将剩余数据中的缺失值替换为指定值
        
        Args:
            seg_signals: 段信号字典
            seg_label: 段标签字典  
            segment_log_id: 用于日志的段标识符
            nan_replacement_value: 替换NaN的值，默认-1.0
            discard_threshold: 丢弃窗口的缺失值比例阈值，默认0.5(50%)
            
        Returns:
            (处理后的信号字典, 处理后的标签字典) 或 None如果所有窗口都被丢弃
        """
        # 1. 获取第一个信号的窗口数量作为基准
        first_signal_name = next(iter(seg_signals.keys()))
        first_signal_dict = seg_signals[first_signal_name]
        first_signal_info = first_signal_dict.get('info', {})
        reference_windows = first_signal_info.get('windows', [])
        num_windows = len(reference_windows)
        
        if num_windows == 0:
            if self.verbose:
                print(f"No windows found in {segment_log_id}")
            return seg_signals, seg_label  # 没有窗口就直接返回原始数据
        
        # 2. 记录需要删除的窗口索引
        windows_to_discard = set()
        
        # 遍历所有信号，检查每个窗口的缺失值比例
        for signal_name, signal_dict in seg_signals.items():
            signal_data = signal_dict.get('data')
            signal_info = signal_dict.get('info', {})
            windows = signal_info.get('windows', reference_windows)
            
            if signal_data is None or signal_data.size == 0:
                continue
                
            # 确保窗口数量一致
            if len(windows) != num_windows:
                if self.verbose:
                    print(f"Warning: {signal_name} has {len(windows)} windows, expected {num_windows}. Using reference windows.")
                windows = reference_windows
            
            # 检查每个窗口的缺失值比例
            for window_idx, window in enumerate(windows):
                start_idx = window.get('start', 0)
                end_idx = window.get('end', signal_data.shape[-1] if signal_data.ndim > 1 else len(signal_data))
                
                # 确保索引有效
                window_data = signal_data[..., start_idx:end_idx]
                if window_data.size == 0:
                    continue
                    
                # 计算窗口内的缺失值
                nan_mask = np.isnan(window_data)
                num_nan = np.sum(nan_mask)
                window_size = window_data.size
                nan_percentage = (num_nan / window_size) if window_size > 0 else 0
                
                if nan_percentage > discard_threshold:
                    windows_to_discard.add(window_idx)
                    if self.verbose:
                        print(f"Marking window {window_idx} of {signal_name} for discard in {segment_log_id}: "
                            f"NaN percentage {nan_percentage*100:.2f}% > {discard_threshold*100}%")
        
        # 3. 检查是否所有窗口都被标记为丢弃
        if len(windows_to_discard) == num_windows:
            if self.verbose:
                print(f"All windows marked for discard in {segment_log_id}")
            return None, None  # 所有窗口都被丢弃，返回None
        
        if self.verbose and windows_to_discard:
            print(f"Discarding {len(windows_to_discard)} out of {num_windows} windows in {segment_log_id}: {sorted(windows_to_discard)}")
        
        # 4. 计算保留的窗口索引
        valid_window_indices = [i for i in range(num_windows) if i not in windows_to_discard]
        
        # 5. 更新所有信号的窗口信息
        processed_signals = {}
        for signal_name, signal_dict in seg_signals.items():
            signal_data = signal_dict.get('data')
            signal_info = signal_dict.get('info', {})
            windows = signal_info.get('windows', reference_windows)
            
            # 创建处理后的信号字典副本
            processed_signal_dict = signal_dict.copy()
            processed_signal_dict['info'] = signal_info.copy()
            
            # 确保窗口数量一致
            if len(windows) != num_windows:
                windows = reference_windows
            
            # 直接根据保留的索引更新窗口列表
            valid_windows = [windows[i] for i in valid_window_indices]
            processed_signal_dict['info']['windows'] = valid_windows
            
            # 替换信号数据中的NaN值
            if signal_data is not None and signal_data.size > 0 and np.isnan(signal_data).any():
                processed_signal_dict['data'] = np.nan_to_num(signal_data, nan=nan_replacement_value)
                if self.verbose:
                    print(f"Replaced NaNs with {nan_replacement_value} in {signal_name} of {segment_log_id}")
            
            processed_signals[signal_name] = processed_signal_dict
        
        # 6. 更新标签的窗口信息
        processed_labels = {}
        for label_name, label_dict in seg_label.items():
            if isinstance(label_dict, dict) and 'info' in label_dict:
                label_info = label_dict.get('info', {})
                label_windows = label_info.get('windows', reference_windows)
                
                processed_label_dict = label_dict.copy()
                processed_label_dict['info'] = label_info.copy()
                
                # 确保标签窗口数量一致
                if len(label_windows) != num_windows:
                    label_windows = reference_windows
                
                # 直接根据保留的索引更新窗口列表
                valid_label_windows = [label_windows[i] for i in valid_window_indices]
                processed_label_dict['info']['windows'] = valid_label_windows
                processed_labels[label_name] = processed_label_dict
            else:
                # 标签没有窗口结构，直接复制
                processed_labels[label_name] = label_dict
        
        if self.verbose:
            print(f"Successfully processed {segment_log_id}: kept {len(valid_window_indices)} out of {num_windows} windows")
        
        return processed_signals, processed_labels


# from dataset.osausd_dataset import OSAUSDDataset
# from dataset.transform import Filter, SlideWindow, Crop, Select, PickChannels,Mapping, MinMaxNormalize,Reshape

# offline_signal_transform = [
#     PickChannels(channels=['II'], source='ecg', target='ecg'),
#     # Filter(l_freq=5.0, h_freq=35.0, method='iir', iir_params=dict(order=2, ftype='butter'), source='ecg', target='ecg'),
#     SlideWindow(window_size=180*80, stride=60*80, source='ecg', target='ecg'),
#     SlideWindow(window_size=180, stride=60, source='spo2', target='spo2'),
#     Select(key=['ecg', 'spo2']),
# ]

# offline_label_transform = [
#     Crop(crop_left=60, crop_right=60, source='anomaly', target='anomaly'),
#     SlideWindow(window_size=60, stride=60, source='anomaly', target='anomaly'),
#     Select(key=['anomaly']),

# ]
# online_signal_transform = [
#     MinMaxNormalize(source='ecg', target='ecg'),
#     MinMaxNormalize(source='spo2', target='spo2'),
#     Reshape(shape=(180,1), source='spo2', target='spo2'),
# ]
# online_label_transform = [
#     Mapping(mapping={
#         True: 1,
#         False: -1,
#     }, source='anomaly', target='anomaly'),
# ]

# dataset = OSAUSDDataset(
#     root_path='/mnt/ssd/lingyus/Stroke_Unit_dataset/dataset_OSAS.pickle',
#     io_path='/mnt/ssd/lingyus/tyee_osausd/train',
#     # io_chunks=224,
#     offline_label_transform=offline_label_transform,
#     offline_signal_transform=offline_signal_transform,
#     online_signal_transform=online_signal_transform,
#     online_label_transform=online_label_transform,
#     io_mode='hdf5',
#     io_chunks=180*80
#     # num_worker=4,
# )