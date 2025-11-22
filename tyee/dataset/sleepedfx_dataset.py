#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : sleepedfx_dataset.py
@Time    : 2024/12/25 15:36:19
@Desc    : 
"""


import os
import mne
import torch
import numpy as np
from tyee.dataset import BaseDataset
from typing import Any, Callable, Union, Dict, List, Tuple, Generator

def list_records(root_path):
    recording_files = []
    scoring_files = []

    files = os.listdir(root_path)

    for file in files:
        if 'PSG' in file:
            recording_files.append(os.path.join(root_path, file))
        if 'Hypnogram' in file:
            scoring_files.append(os.path.join(root_path, file))

    recording_files.sort()
    scoring_files.sort()

    recording_scoring_pairs = []
    for recording_file, scoring_file in zip(recording_files, scoring_files):
        if recording_file[:6] == scoring_file[:6]:
            recording_scoring_pairs.append((recording_file, scoring_file))

    return recording_scoring_pairs


# telemetry ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Marker']
# cassette ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
class SleepEDFCassetteDataset(BaseDataset):
    def __init__(
        self,
        # Cassette dataset parameters
        crop_wake_mins: int = 30,
        # common parameters
        root_path: str = './sleep-edf-database-expanded-1.0.0/sleep-cassette',
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
        self.crop_wake_mins = crop_wake_mins
        super().__init__(**params)

    def set_records(self, root_path, **kwargs):
        recording_scoring_pairs = []

        recording_scoring_pairs = list_records(root_path)

        return recording_scoring_pairs

    def read_record(self, record: Tuple, **kwargs) -> Dict:
        recording_file, scoring_file = record

        raw = mne.io.read_raw_edf(recording_file, preload=True)
        annotation = mne.read_annotations(scoring_file)
        raw.set_annotations(annotation, emit_warning=False)
        data = raw.get_data(units='uV')
        channels = ['Fpz-Cz', 'Pz-Oz', 'horizontal', 'oro-nasal', 'submental', 'rectal']
        if self.crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annotation.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annotation[int(sleep_event_inds[0])]["onset"] - self.crop_wake_mins * 60
            tmax = annotation[int(sleep_event_inds[-1])]["onset"] + self.crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))
        print(data.shape)
        freq = raw.info['sfreq']
        eeg = {
            'data': data[0:2, :],
            'freq': freq,
            'channels': ['Fpz-Cz', 'Pz-Oz']
        }
        eog = {
            'data': data[2:3, :],
            'freq': freq,
            'channels': ['horizontal']
        }
        rsp = {
            'data': data[3:4, :],
            'freq': freq,
            'channels': ['oro-nasal']
        }
        emg = {
            'data': data[4:5, :],
            'freq': freq,
            'channels': ['submental']
        }
        temp = {
            'data': data[5:6, :],
            'freq': freq,
            'channels': ['rectal']
        }
        events, event_id = mne.events_from_annotations(
            raw, chunk_duration=30.)
        print(f"Number of events: {len(events)}")
        print(events)
        if 'Sleep stage ?' in event_id.keys():
            event_id.pop('Sleep stage ?')
        if 'Movement time' in event_id.keys():
            event_id.pop('Movement time')
        stage_mapping = {
            'W': 0,  # Wake
            '1': 1,  # NREM Stage 1
            '2': 2,  # NREM Stage 2
            '3': 3,  # NREM Stage 3
            '4': 4,  # NREM Stage 4
            'R': 5,  # REM
        }
        id_to_event = {v: k for k, v in event_id.items()}
        segments = []
        for i in range(len(events)):
            stim = events[i][2]
            if stim not in id_to_event:
                continue
            start = events[i][0] / freq
            end = start + 30.0
            stage_char = str(id_to_event[stim][-1])
            stage_numeric = stage_mapping.get(stage_char, -1)
            segments.append({
                'start': start,
                'end': end,
                'value': {
                    'stage': {
                        'data': stage_numeric
                    }
                }
            })
        return {
            'signals': {
                'eeg': eeg,
                'eog': eog,
                'rsp': rsp,
                'emg': emg,
                'temp': temp
            },
            'labels': {
                'segments': segments,
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(recording_file))[0],
            }
        }
        
    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        Segment all signal types in signals according to label['segments'], and return signals and labels for each segment.
        The start/end units in label['segments'] are in seconds.
        """
        stacked_signals = {}
        stacked_labels = {}
        valid_segments_info = []
        segments = []
        for sig_type in signals.keys():
            stacked_signals[sig_type] = []
        
        if len(labels['segments']) > 0:
            sample_labels = labels['segments'][0]['value']
            for label_type in sample_labels.keys():
                stacked_labels[label_type] = []
        
        for seg_idx, seg in enumerate(labels['segments']):
            start_time = seg['start']
            end_time = seg['end']
            
            segment_valid = True
            temp_seg_signals = {}
            
            for sig_type, sig in signals.items():
                freq = sig['freq']
                data = sig['data']
                start_idx = int(round(start_time * freq))
                end_idx = int(round(end_time * freq))
                
                if self.include_end:
                    end_idx += 1
                    
                if start_idx < 0 or end_idx > data.shape[-1] or start_idx >= end_idx:
                    print(f"Invalid segment {seg_idx}: {sig_type}, {start_time}-{end_time}s, "
                        f"indices {start_idx}-{end_idx}, data_shape={data.shape}")
                    segment_valid = False
                    break
                
                temp_seg_signals[sig_type] = {
                    'data': data[..., start_idx:end_idx],
                    'channels': sig.get('channels', []),
                    'freq': freq,
                }
            
            if segment_valid:
                for sig_type, seg_signal in temp_seg_signals.items():
                    stacked_signals[sig_type].append(seg_signal['data'])
                
                for label_type, label_info in seg['value'].items():
                    if 'data' in label_info:
                        stacked_labels[label_type].append(label_info['data'])
                
                valid_segments_info.append({
                    'segment_idx': seg_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                })
                
                # print(f"Added segment {seg_idx}: {start_time:.1f}-{end_time:.1f}s")
        
        final_stacked_signals = {}
        for sig_type, signal_list in stacked_signals.items():
            if len(signal_list) > 0:
                stacked_data = np.stack(signal_list, axis=0)
                final_stacked_signals[sig_type] = {
                    'data': stacked_data,
                    'channels': signals[sig_type].get('channels', []),
                    'freq': signals[sig_type]['freq'],
                }
                print(f"Stacked {sig_type}: {len(signal_list)} segments -> {stacked_data.shape}")
        
        final_stacked_labels = {}
        for label_type, label_list in stacked_labels.items():
            if len(label_list) > 0:
                stacked_label_data = np.stack(label_list, axis=0)
                final_stacked_labels[label_type] = {
                    'data': stacked_label_data
                }
                print(f"Stacked {label_type}: {len(label_list)} segments -> {stacked_label_data.shape}")
        
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
        return file_name.split('-')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'

# telemetry ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Marker']
class SleepEDFTelemetryDataset(BaseDataset):
    def __init__(
        self,
        # Cassette dataset parameters
        crop_wake_mins: int = 30,
        # common parameters
        root_path: str = './sleep-edf-database-expanded-1.0.0/sleep-telemetry',
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
        self.crop_wake_mins = crop_wake_mins
        super().__init__(**params)

    def set_records(self, root_path, **kwargs):
        recording_scoring_pairs = []

        recording_scoring_pairs = list_records(root_path)

        return recording_scoring_pairs

    def read_record(self, record: Tuple, **kwargs) -> Dict:
        recording_file, scoring_file = record

        raw = mne.io.read_raw_edf(recording_file, preload=True)
        annotation = mne.read_annotations(scoring_file)
        raw.set_annotations(annotation, emit_warning=False)
        data = raw.get_data(units='uV')
        channels = ['Fpz-Cz', 'Pz-Oz', 'horizontal', 'submental']
        if self.crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annotation.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annotation[int(sleep_event_inds[0])]["onset"] - self.crop_wake_mins * 60
            tmax = annotation[int(sleep_event_inds[-1])]["onset"] + self.crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))
        print(data.shape)
        freq = raw.info['sfreq']
        eeg = {
            'data': data[0:2, :],
            'freq': freq,
            'channels': ['Fpz-Cz', 'Pz-Oz']
        }
        eog = {
            'data': data[2:3, :],
            'freq': freq,
            'channels': ['horizontal']
        }
        emg = {
            'data': data[3:4, :],
            'freq': freq,
            'channels': ['submental']
        }
        events, event_id = mne.events_from_annotations(
            raw, chunk_duration=30.)
        print(f"Number of events: {len(events)}")
        print(events)
        if 'Sleep stage ?' in event_id.keys():
            event_id.pop('Sleep stage ?')
        if 'Movement time' in event_id.keys():
            event_id.pop('Movement time')
        stage_mapping = {
            'W': 0,  # Wake
            '1': 1,  # NREM Stage 1
            '2': 2,  # NREM Stage 2
            '3': 3,  # NREM Stage 3
            '4': 4,  # NREM Stage 4
            'R': 5,  # REM
        }
        id_to_event = {v: k for k, v in event_id.items()}
        segments = []
        for i in range(len(events)):
            stim = events[i][2]
            if stim not in id_to_event:
                continue
            start = events[i][0] / freq
            end = start + 30.0
            stage_char = str(id_to_event[stim][-1])
            stage_numeric = stage_mapping.get(stage_char, -1)
            segments.append({
                'start': start,
                'end': end,
                'value': {
                    'stage': {
                        'data': stage_numeric
                    }
                }
            })
        return {
            'signals': {
                'eeg': eeg,
                'eog': eog,
                'emg': emg,
            },
            'labels': {
                'segments': segments,
            },
            'meta': {
                'file_name': os.path.splitext(os.path.basename(recording_file))[0],
            }
        }
        
    def segment_split(
        self,
        signals: Dict[str, Any],
        labels: Dict[str, Any],
    ) -> list:
        """
        Segment all signal types in signals according to label['segments'], and return signals and labels for each segment.
        The start/end units in label['segments'] are in seconds.
        """
        stacked_signals = {}
        stacked_labels = {}
        valid_segments_info = []
        segments = []
        for sig_type in signals.keys():
            stacked_signals[sig_type] = []
        
        if len(labels['segments']) > 0:
            sample_labels = labels['segments'][0]['value']
            for label_type in sample_labels.keys():
                stacked_labels[label_type] = []
        
        for seg_idx, seg in enumerate(labels['segments']):
            start_time = seg['start']
            end_time = seg['end']
            
            segment_valid = True
            temp_seg_signals = {}
            
            for sig_type, sig in signals.items():
                freq = sig['freq']
                data = sig['data']
                start_idx = int(round(start_time * freq))
                end_idx = int(round(end_time * freq))
                
                if self.include_end:
                    end_idx += 1
                    
                if start_idx < 0 or end_idx > data.shape[-1] or start_idx >= end_idx:
                    print(f"Invalid segment {seg_idx}: {sig_type}, {start_time}-{end_time}s, "
                        f"indices {start_idx}-{end_idx}, data_shape={data.shape}")
                    segment_valid = False
                    break
                
                temp_seg_signals[sig_type] = {
                    'data': data[..., start_idx:end_idx],
                    'channels': sig.get('channels', []),
                    'freq': freq,
                }
            
            if segment_valid:
                for sig_type, seg_signal in temp_seg_signals.items():
                    stacked_signals[sig_type].append(seg_signal['data'])
                
                for label_type, label_info in seg['value'].items():
                    if 'data' in label_info:
                        stacked_labels[label_type].append(label_info['data'])
                
                valid_segments_info.append({
                    'segment_idx': seg_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                })
                
                # print(f"Added segment {seg_idx}: {start_time:.1f}-{end_time:.1f}s")
        
        final_stacked_signals = {}
        for sig_type, signal_list in stacked_signals.items():
            if len(signal_list) > 0:
                stacked_data = np.stack(signal_list, axis=0)
                final_stacked_signals[sig_type] = {
                    'data': stacked_data,
                    'channels': signals[sig_type].get('channels', []),
                    'freq': signals[sig_type]['freq'],
                }
                print(f"Stacked {sig_type}: {len(signal_list)} segments -> {stacked_data.shape}")
        
        final_stacked_labels = {}
        for label_type, label_list in stacked_labels.items():
            if len(label_list) > 0:
                stacked_label_data = np.stack(label_list, axis=0)
                final_stacked_labels[label_type] = {
                    'data': stacked_label_data
                }
                print(f"Stacked {label_type}: {len(label_list)} segments -> {stacked_label_data.shape}")
        
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
        return file_name.split('-')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        return f'{idx}_{file_name}'



# from dataset.sleepedfx_dataset import SleepEDFCassetteDataset
# from dataset.transform import SlideWindow, Select, PickChannels, Mapping, Transpose, Reshape, ExpandDims, Compose

# before_segment_transform =[
#     PickChannels(channels=['Fpz-Cz'], source='eeg', target='eeg'),
# ]
# offline_signal_transform = [
#     SlideWindow(window_size=20, stride=20, axis=0, source='eeg', target='eeg'),
#     SlideWindow(window_size=20, stride=20, axis=0, source='eog', target='eog'),
#     Select(key=['eeg', 'eog']),
# ]
# offline_label_transform = [
#     Mapping(
#         mapping={
#             0:0,  # Sleep stage W
#             1:1,  # Sleep stage N1
#             2:2,  # Sleep stage N2
#             3:3,  # Sleep stage N3
#             4:3, # Sleep stage N4
#             5:4,  # Sleep stage R
#         },source='stage', target='stage'),
#     SlideWindow(window_size=20, stride=20, axis=0, source='stage', target='stage'),
# ]
# online_signal_transform = [
#     Compose(transforms=[
#         Transpose(axes=(1,0,2)),
#         Reshape(shape=(1, -1)),
#         ExpandDims(axis=-1)],source='eeg', target='eeg'),
#     Compose(transforms=[
#         Transpose(axes=(1,0,2)),
#         Reshape(shape=(1, -1)),
#         ExpandDims(axis=-1)],source='eog', target='eog'),
# ]
# dataset = SleepEDFCassetteDataset(
#     root_path='/mnt/ssd/lingyus/sleep-edf-20',
#     # root_path='/mnt/ssd/lingyus/test',
#     io_path='/mnt/ssd/lingyus/tyee_sleepedfx_20/train',
#     io_mode='hdf5',
#     before_segment_transform=before_segment_transform,
#     offline_signal_transform=offline_signal_transform,
#     offline_label_transform=offline_label_transform,
#     online_signal_transform=online_signal_transform,
#     io_chunks=20,
#     crop_wake_mins=30,
#     num_worker=8,
# )