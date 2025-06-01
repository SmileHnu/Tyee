#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : cinc2018_dataset.py
@Time    : 2025/05/30 19:26:11
@Desc    : 
"""


import os
import h5py
import glob
import mne
import pandas as pd
import numpy as np
import scipy.io as scio
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class CinC2018Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './challenge-2018/training',
        start_offset: float = 0.0,
        end_offset: float = 0.0,
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
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'
        subjects = os.listdir(root_path)
        subjects_path = [os.path.join(root_path, subject) for subject in subjects]
        edf_events_files_pruned = []
        for subject_path in subjects_path:
            if "RECORDS" in subject_path or "ANNOTATORS" in subject_path or len(os.listdir(subject_path)) != 4:
                continue

            files = glob.glob(f'{subject_path}/*')
            try:
                temp_dict = {}
                for file in files:
                    if '.hea' in file:
                        temp_dict["hea"] = os.path.join(file)
                    elif '-arousal.mat' in file:
                        temp_dict["arousal_mat"] = os.path.join(file)
                    elif 'mat' in file:
                        temp_dict["mat"] = os.path.join(file)
                edf_events_files_pruned.append(temp_dict)
            except:
                continue
        return edf_events_files_pruned
    
    def read_record(self, record: dict, **kwargs) -> Dict:
        arousal_mat_filename = record["arousal_mat"]
        hea_filename = record["hea"]
        edf_filename = record["mat"]
        file_name = edf_filename.split("/")[-1].split(".")[0]
        try:
            channels, Fs, n_samples = self.import_signal_names(hea_filename)
            print(f"Channels: {channels}, Sampling Frequency: {Fs}, Number of Samples: {n_samples}")
            labels, label_names = self.extract_labels(arousal_mat_filename)
            data = scio.loadmat(edf_filename)['val']
            print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
            # 数据切割和对齐
            diff = np.diff(labels, axis=0)
            cutoff = np.where(diff[:, 4] != 0)[0] + 1  # 找到标签变化的索引
            data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]
            info = mne.create_info(channels, Fs, ch_types = 'eeg')
            edf_raw = mne.io.RawArray(data, info)
            data = edf_raw.get_data()
            # 信号模态分割 
            # EEG 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'
            eeg = {
                'data': data[:6],
                'channels': channels[:6],
                'freq': Fs,
            }
            # EOG 'E1-M2'
            eog = {
                'data': data[6:7],
                'channels': channels[6:7],
                'freq': Fs,
            }
            # EMG 'Chin1-Chin2'
            emg = {
                'data': data[7:8],
                'channels': channels[7:8],
                'freq': Fs,
            }
            # 呼吸信号 'ABD', 'CHEST', 'AiRFLOW', 'SaO2'
            # 这里的 'ABD' 是腹部呼吸，'CHEST' 是胸部呼吸，'AiRFLOW' 是气流，'SaO2' 是血氧饱和度
            abd = {
                'data': data[8:9],
                'channels': channels[8:9],
                'freq': Fs,
            }
            # 
            chest = {
                'data': data[9:10],
                'channels': channels[9:10],
                'freq': Fs,
            }
            airflow = {
                'data': data[10:11],
                'channels': channels[10:11],
                'freq': Fs,
            }
            sao2 = {
                'data': data[11:12],
                'channels': channels[11:12],
                'freq': Fs,
            }
            ecg = {
                'data': data[12:13],
                'channels': channels[12:13],
                'freq': Fs,
            }
            # 睡眠事件处理
            EVENT_TO_ID = {
                "wake": 1, 
                "nonrem1": 2, 
                "nonrem2": 3, 
                "nonrem3": 4, 
                "rem": 5, 
            }
            ID_TO_EVENT = {value: key for key, value in EVENT_TO_ID.items()}
            events = self.process_labels_to_events(labels, label_names)
            label_dict = dict(zip(np.arange(0,6), label_names))
            f = lambda x: label_dict[x]
            events = np.array(events)
            annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))
            edf_raw.set_annotations(annotations, emit_warning=True)
            events_raw, _ = mne.events_from_annotations(edf_raw, event_id=EVENT_TO_ID, chunk_duration=30.)
            event_ids = set(events_raw[:, 2])
            segments = []
            for i in range(len(events_raw)):
                stim = events_raw[i][2]
                if stim not in ID_TO_EVENT:
                    continue
                start = events_raw[i][0] / Fs
                end = start + 30.0
                segments.append({
                    'start': start,
                    'end': end,
                    'value': {
                        'stage': {
                            'data': int(stim-1),  # Convert to zero-based index
                        }
                    }
                })

            return {
                'signals': {
                    'eeg': eeg,
                    'eog': eog,
                    'emg': emg,
                    'abd': abd,
                    'chest': chest,
                    'airflow': airflow,
                    'sao2': sao2,
                    'ecg': ecg,
                },
                'labels': {
                    'segments': segments,
                },
                'meta': {
                    'file_name': file_name,
                }
            }
        except Exception as e:
            print(f"Warning: An error occurred {file_name}- {e}")
            return {
                'signals': None,
                'labels': None,
                'meta': {
                    'file_name': file_name,
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
                'trial_id': self.get_trial_id(),
            })
            yield self.assemble_segment(
                key=segment_id,
                signals=seg_signals,
                labels=seg_label,
                info=seg_info,
            )
    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        对 signals 中所有信号类型按 label['segments'] 分段，将所有段按信号类型和标签类型分别堆叠，
        返回堆叠后的信号和标签。
        label['segments'] 的 start/end 单位为秒。
        """
        # 存储每种信号类型的所有段
        stacked_signals = {}
        stacked_labels = {}
        valid_segments_info = []
        segments = []
        # 初始化每种信号类型的列表
        for sig_type in signals.keys():
            stacked_signals[sig_type] = []
        
        # 初始化标签类型的列表（从第一个segment中获取标签类型）
        if len(labels['segments']) > 0:
            sample_labels = labels['segments'][0]['value']
            for label_type in sample_labels.keys():
                stacked_labels[label_type] = []
        
        for seg_idx, seg in enumerate(labels['segments']):
            start_time = seg['start']
            end_time = seg['end']
            
            # 检查这个段是否对所有信号类型都有效
            segment_valid = True
            temp_seg_signals = {}
            
            for sig_type, sig in signals.items():
                freq = sig['freq']
                data = sig['data']
                start_idx = int(round(start_time * freq))
                end_idx = int(round(end_time * freq))
                
                if self.include_end:
                    end_idx += 1
                    
                # 检查索引是否有效
                if start_idx < 0 or end_idx > data.shape[-1] or start_idx >= end_idx:
                    print(f"Invalid segment {seg_idx}: {sig_type}, {start_time}-{end_time}s, "
                        f"indices {start_idx}-{end_idx}, data_shape={data.shape}")
                    segment_valid = False
                    break
                
                # 临时存储这个段的信号
                temp_seg_signals[sig_type] = {
                    'data': data[..., start_idx:end_idx],
                    'channels': sig.get('channels', []),
                    'freq': freq,
                }
            
            # 如果这个段对所有信号类型都有效，则添加到堆叠列表中
            if segment_valid:
                # 添加信号段到各自的列表
                for sig_type, seg_signal in temp_seg_signals.items():
                    stacked_signals[sig_type].append(seg_signal['data'])
                
                # 添加标签段到各自的列表
                for label_type, label_info in seg['value'].items():
                    if 'data' in label_info:
                        stacked_labels[label_type].append(label_info['data'])
                
                # 记录段信息
                valid_segments_info.append({
                    'segment_idx': seg_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                })
                
                # print(f"Added segment {seg_idx}: {start_time:.1f}-{end_time:.1f}s")
        
        # 在第一个维度（axis=0）上堆叠每种信号类型
        final_stacked_signals = {}
        for sig_type, signal_list in stacked_signals.items():
            if len(signal_list) > 0:
                # 在第一个维度（axis=0）上堆叠
                stacked_data = np.stack(signal_list, axis=0)
                final_stacked_signals[sig_type] = {
                    'data': stacked_data,
                    'channels': signals[sig_type].get('channels', []),
                    'freq': signals[sig_type]['freq'],
                }
                print(f"Stacked {sig_type}: {len(signal_list)} segments -> {stacked_data.shape}")
        
        # 在第一个维度（axis=0）上堆叠每种标签类型
        final_stacked_labels = {}
        for label_type, label_list in stacked_labels.items():
            if len(label_list) > 0:
                # 在第一个维度（axis=0）上堆叠
                stacked_label_data = np.stack(label_list, axis=0)
                final_stacked_labels[label_type] = {
                    'data': stacked_label_data
                }
                print(f"Stacked {label_type}: {len(label_list)} segments -> {stacked_label_data.shape}")
        
        # 计算总时长
        total_duration = sum([info['duration'] for info in valid_segments_info])
        
        segments.append({
            'signals': final_stacked_signals,
            'labels': final_stacked_labels,
            'info': {
                'num_epoch': len(valid_segments_info),
                'total_duration': total_duration,
            }
        })
        return segments
    
    def get_subject_id(self, file_name) -> str:
        # Extract the subject ID from the file name
        # Assuming the file name format is like "subject_id_record_id.edf"
        # You can modify this logic based on your actual file naming convention
        return file_name
    
    def get_segment_id(self, file_name, idx) -> str:
        # Extract the segment ID from the file name
        # Assuming the segment ID is the same as the file name in this case
        return f'{idx}_{file_name}'
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        # Extract the sample ID from the file name and index
        # Assuming the sample ID is a combination of the file name and index
        return [f"{i}_{segment_id}" for i in range(sample_len)]
    
    def import_signal_names(self, file_name):
        with open(file_name, 'r') as myfile:
            s = myfile.read()
            s = s.split('\n')
            s = [x.split() for x in s]

            n_signals = int(s[0][1])
            n_samples = int(s[0][3])
            Fs        = int(s[0][2])

            s = s[1:-1]
            s = [s[i][8] for i in range(0, n_signals)]
        return s, Fs, n_samples

    def extract_labels(self, path):
        data = h5py.File(path, 'r')
        length = data['data']['sleep_stages']['wake'].shape[1]
        labels = np.zeros((length, 6)) 

        for i, label in enumerate(data['data']['sleep_stages'].keys()):
            labels[:,i] = data['data']['sleep_stages'][label][:]
        
        return labels, list(data['data']['sleep_stages'].keys())

    def process_labels_to_events(self, labels, label_names):
        new_labels = np.argmax(labels, axis = 1)
        lab = new_labels[0]
        events = []
        start = 0
        i = 0
        while i < len(new_labels)-1:
            while new_labels[i] == lab and i < len(new_labels)-1:
                i+=1
            end = i
            dur = end +1 - start
            events.append([start, dur, lab])
            lab = new_labels[i]
            start = i+1
        return events


# from dataset.cinc2018_dataset import CinC2018Dataset
# from dataset.transform import SlideWindow, Resample,Compose,Select,Concat,Squeeze,PickChannels
# before_segment_transform = [
#    PickChannels(channels=["C3-M2", "C4-M1", "O1-M2", "O2-M1"], source='eeg', target='eeg'),
#     Concat(axis=0, source=['eeg', 'eog'], target='ss'),
#     Concat(axis=0, source=['chest','sao2','abd'], target='resp'),
#     Resample(desired_freq=256,source='ss', target='ss'),
#     Resample(desired_freq=256,source='resp', target='resp'),
#     Resample(desired_freq=256,source='ecg', target='ecg'),
#     Select(key=['ss','resp','ecg']),
# ]
# offline_signal_transform = [
#     SlideWindow(window_size=1, stride=1, axis=0, source='ss', target='ss'),
#     SlideWindow(window_size=1, stride=1, axis=0, source='resp', target='resp'),
#     SlideWindow(window_size=1, stride=1, axis=0, source='ecg', target='ecg'),
# ]
# offline_label_transform = [
#     SlideWindow(window_size=1, stride=1, axis=0, source='stage', target='stage'),
# ]
# online_signal_transform = [
#     Squeeze(axis=0, source='ss', target='ss'),
#     Squeeze(axis=0, source='resp', target='resp'),
#     Squeeze(axis=0, source='ecg', target='ecg'),
# ]
# online_label_transform = [
#     Squeeze(axis=0, source='stage', target='stage'),
# ]

# # 基础路径配置
# base_root_path = '/mnt/ssd/lingyus/challenge-2018-split'
# base_io_path = '/mnt/ssd/lingyus/tyee_cinc2018'

# # 创建三个数据集
# splits = ['train', 'valid', 'test']
# datasets = {}

# for split in splits:
#     root_path = f'{base_root_path}/{split}'
#     io_path = f'{base_io_path}/{split}'
    
#     print(f"Creating {split} dataset...")
#     print(f"  Root path: {root_path}")
#     print(f"  IO path: {io_path}")
    
#     datasets[split] = CinC2018Dataset(
#         root_path=root_path,
#         io_path=io_path,
#         io_mode='hdf5',
#         io_chunks=1,
#         before_segment_transform=before_segment_transform,
#         offline_signal_transform=offline_signal_transform,
#         offline_label_transform=offline_label_transform,
#         online_signal_transform=online_signal_transform,
#         online_label_transform=online_label_transform,
#         num_worker=8,
#     )