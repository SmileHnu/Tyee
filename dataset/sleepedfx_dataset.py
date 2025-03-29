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
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, List, Tuple
from dataset.constants.standard_channels import EEG_CHANNELS_ORDER

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

def interploate(raw: mne.io.BaseRaw, channels: List = ['EEG Fpz-Cz', 'EEG Pz-Oz']):
    missing_channels = list(set(channels) - set(raw.ch_names))
    if missing_channels:
        info = mne.create_info(missing_channels, raw.info['sfreq'], 'eeg')
        zero_data = np.zeros((len(missing_channels), len(raw.times)))
        raw_missing = mne.io.RawArray(zero_data, info)
        raw.add_channels([raw_missing], force_update_info=True)

    # raw = raw.pick_channels(channels)
    return raw

def filter(raw: mne.io.BaseRaw, l_freq: float = 0.5, h_freq: float = 30):
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    return raw


def downsample(epochs: mne.Epochs, sfreq: int = 100):
    epochs = epochs.resample(sfreq)
    return epochs

# telemetry ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Marker']
# cassette ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker']
class SleepEDFxDataset(BaseDataset):
    def __init__(self,
                 root_path: str = './sleep-edf-database-expanded-1.0.0/',
                 studies: List = ['cassette', 'telemetry'],
                 channels: List = ['EEG Fpz-Cz',
                                   'EEG Pz-Oz'],
                signal_types: List = ['eeg','eog','emg','rsp','temp'],
                 l_freq: float = 0.5,
                 h_freq: float = 30,
                 sfreq: int = 100,
                 crop_wake_mins: int = 30,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')
        print(studies)
        assert 'cassette' in studies or 'telemetry' in studies, 'studies must contain either "cassette" or "telemetry"'

        params = {
            'root_path': root_path,
            'studies': studies,
            'channels': channels,
            'signal_types': signal_types,
            'l_freq': l_freq,
            'h_freq': h_freq,
            'sfreq': sfreq,
            'crop_wake_mins': crop_wake_mins,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        self.__dict__.update(params)
        super().__init__(**params)
        # save all arguments to __dict__
    def set_records(self, root_path, **kwargs):
        recording_scoring_pairs = []

        if 'cassette' in self.studies:
            recording_scoring_pairs += list_records(
                os.path.join(root_path, 'sleep-cassette'))

        if 'telemetry' in self.studies:
            recording_scoring_pairs += list_records(
                os.path.join(root_path, 'sleep-telemetry'))

        return recording_scoring_pairs

    @staticmethod
    def read_record(record: Tuple,
                    channels: List = ['EEG Fpz-Cz',
                                      'EEG Pz-Oz'],
                    l_freq: float = 0.5,
                    h_freq: float = 30,
                    sfreq: int = 100,
                    crop_wake_mins: int = 30,
                     **kwargs) -> Dict:
        recording_file, scoring_file = record

        raw = mne.io.read_raw_edf(recording_file, preload=True)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        raw = filter(raw, l_freq, h_freq)
        raw = downsample(raw, sfreq)
        raw = interploate(raw, channels)
        annotation = mne.read_annotations(scoring_file)
        raw.set_annotations(annotation, emit_warning=False)
        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annotation.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annotation[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annotation[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))
        events, event_id = mne.events_from_annotations(
            raw, chunk_duration=30.)

        if 'Sleep stage ?' in event_id.keys():
            event_id.pop('Sleep stage ?')
        if 'Movement time' in event_id.keys():
            event_id.pop('Movement time')

        tmax = 30. - 1. / raw.info['sfreq']
        epochs = mne.Epochs(raw=raw, events=events,
                            event_id=event_id, tmin=0., tmax=tmax, baseline=None)

        epochs_data = epochs.get_data(units='uV')
        print(epochs_data.shape)
        epochs_channel = epochs.ch_names
        epochs_label = []
        for epoch_annotation in epochs.get_annotations_per_epoch():
            epochs_label.append(epoch_annotation[0][2])

        return {
            'epochs_data': epochs_data,
            'epochs_label': epochs_label,
            'epochs_channel': epochs_channel
        }
        
    @staticmethod
    def process_record(record: Tuple,
                       result: Dict,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        epochs_data = result['epochs_data'].copy()
        epochs_label = result['epochs_label'].copy()
        epoch_channel = result['epochs_channel'].copy()
        recording_file, scoring_file = record
        basename = os.path.splitext(os.path.basename(recording_file))[0]
        subject_id = basename[3:5]
        session_id = basename[5]
        # if before_trial:
        #     epochs_data = before_trial(epochs_data)

        label2id = {'Sleep stage W': 0,
                    'Sleep stage 1': 1,
                    'Sleep stage 2': 2,
                    'Sleep stage 3': 3,
                    'Sleep stage 4': 3,
                    'Sleep stage R': 4}

        for i, (epoch_data, epoch_label) in enumerate(zip(epochs_data, epochs_label)):
            
            result = {}
            if len(epoch_channel) == 7:
                # print('执行了')
                signal_types = ['eeg', 'eog', 'rsp', 'emg', 'temp']
                result = {
                    'eeg': {
                        'signals': epoch_data[0:2],
                        'sampling_rate': 100,
                        'channels': ['EEG Fpz-Cz', 'EEG Pz-Oz']
                    },
                    'eog': {
                        'signals': epoch_data[2:3],
                        'sampling_rate': 100,
                        'channels': ['EOG horizontal']
                    },
                    'rsp': {
                        'signals': epoch_data[3:4],
                        'sampling_rate': 100,
                        'channels': ['Resp oro-nasal']
                    },
                    'emg': {
                        'signals': epoch_data[4:5],
                        'sampling_rate': 100,
                        'channels': ['EMG submental']
                    },
                    'temp': {
                        'signals': epoch_data[5:6],
                        'sampling_rate': 100,
                        'channels': ['Temp rectal']
                    }
                }
            elif len(epoch_channel) == 5:
                signal_types = ['eeg', 'eog', 'emg']
                result = {
                    'eeg': {
                        'signals': epoch_data[0:2],
                        'sampling_rate': 100,
                        'channels': ['EEG Fpz-Cz', 'EEG Pz-Oz']
                    },
                    'eog': {
                        'signals': epoch_data[2:3],
                        'sampling_rate': 100,
                        'channels': ['EOG horizontal']
                    },
                    'emg': {
                        'signals': epoch_data[3:4],
                        'sampling_rate': 100,
                        'channels': ['EMG submental']
                    }
                }
            if not offline_transform is None:
                try:
                    for signal_type in signal_types:
                        if signal_type in offline_transform:
                            result[signal_type] = offline_transform[signal_type](result[signal_type])
                except (KeyError, ValueError) as e:
                    print(f'Error in processing record {basename}: {e}')
                return None

            clip_id = f"{i}_{subject_id}"

            record_info = {
                'clip_id': clip_id,
                'label': label2id[epoch_label],
                'start_at': i * 30,
                'end_at': (i + 1) * 30,
                'subject_id': subject_id,
                'session_id': f'{subject_id}_{session_id}',
                'signal_types': signal_types
            }
            combined_result = {signal_type: result[signal_type] for signal_type in signal_types if signal_type in result}
            # print(combined_result)
            combined_result.update({
                'key': clip_id,
                'info': record_info
            })
            yield combined_result

    def __getitem__(self, index):
        info = self.read_info(index)
        
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])

        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        result['label'] = info['label']
        if self.label_transform is not None:
            result['label'] = self.label_transform(result['label'])
        if self.online_transform is not None:
            for signal_type in self.signal_types:
                if signal_type in self.online_transform:
                    result[signal_type] = self.online_transform[signal_type](result[signal_type])
                    if 'ToIndexChannels' not in [transform.__class__.__name__ for transform in self.online_transform[signal_type].transforms]:
                        if 'channels' in result[signal_type]:
                            del result[signal_type]['channels']
                else:
                    del result[signal_type]['channels']
        else:
            for signal_type in self.signal_types:
                if 'channels' in result[signal_type]:
                    del result[signal_type]['channels']
        return result
