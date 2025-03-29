#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : seedv_dataset.py
@Time    : 2024/12/26 20:12:55
@Desc    : 
"""
import os
import mne
import torch
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union
from dataset.constants.standard_channels import EEG_CHANNELS_ORDER

class SEEDVDataset(BaseDataset):
    def __init__(self,
                 root_path: str = './EEG_raw',
                 chunk_size: int = 2000,
                 overlap: int = 0,
                 num_channel: int = 62,
                 online_transform: Union[None, Callable] = None,
                 offline_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 before_trial: Union[None, Callable] = None,
                 after_trial: Union[Callable, None] = None,
                 after_session: Union[Callable, None] = None,
                 after_subject: Union[Callable, None] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = 1048576,
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 signal_types: list = ['eeg'],
                 verbose: bool = True,
                 ):
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'online_transform': online_transform,
            'offline_transform': offline_transform,
            'label_transform': label_transform,
            'before_trial': before_trial,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'signal_types': signal_types,
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = os.listdir(root_path)
        file_list = [
            os.path.join(root_path, file) for file in file_list
            if file.endswith('.cnt')
        ]
        return file_list
    @staticmethod
    def read_record(record: str, **kwargs):

        try:
            eeg_raw = mne.io.read_raw_cnt(record, preload=True)
        except Exception as e:
            print(f"Error loading {record}: {e}")
            result = None
            return result
        useless_ch = ['M1', 'M2', 'VEO', 'HEO']
        eeg_raw.drop_channels(useless_ch)
        eeg_channels = eeg_raw.ch_names
        sampling_rate = eeg_raw.info['sfreq']
        eeg_signals = eeg_raw.get_data(units='uV')
        print(eeg_signals.shape)
        eeg = {
            'signals': eeg_signals,
            'sampling_rate': sampling_rate,
            'channels': eeg_channels,
        }
        result = {
            'eeg': eeg
        }
        return result
        
    @staticmethod
    def process_record(record, 
                       signal_types,
                       result,
                       offline_transform,
                       num_channel: int = 62,
                       chunk_size: int = 800,
                       overlap: int = 0,
                       **kwargs):
        if result is None:
            return None
        file_name = os.path.splitext(os.path.basename(record))[0]

        print(f'Processing record: {file_name}')

        subject_id, session_id, date = file_name.split('_')[:3]

        labels = [[4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
                  [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
                  [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]]

        trial_labels = labels[int(session_id) - 1]

        start_end_list = [
            {
                'start_seconds': [
                    30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227,
                    2435, 2667, 2932, 3204
                ],
                'end_seconds': [
                    102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401,
                    2607, 2901, 3172, 3359
                ]
            },
            {
                'start_seconds': [
                    30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966,
                    2186, 2333, 2490, 2741
                ],
                'end_seconds': [
                    267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153,
                    2302, 2428, 2709, 2817
                ]
            },
            {
                'start_seconds': [
                    30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055,
                    2307, 2457, 2726, 2888
                ],
                'end_seconds': [
                    321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275,
                    2425, 2664, 2857, 3066
                ]
            },
        ]
        start_seconds = start_end_list[int(session_id) - 1]['start_seconds']
        end_seconds = start_end_list[int(session_id) - 1]['end_seconds']

        write_pointer = 0
        all_signals = result['eeg']['signals'].copy()
        for trial_id, (start_second,
                       end_second) in enumerate(zip(start_seconds,
                                                    end_seconds)):
            print(f'Processing trial: {trial_id}')
            trial_meta_info = {
                'subject_id': subject_id,
                'session_id': f'{subject_id}_{session_id}',
                'date': date,
                'trial_id': f'{subject_id}_{session_id}_{trial_id}',
                'label': trial_labels[trial_id]
            }
            result['eeg']['signals'] = all_signals[:,
                                        start_second * 1000:end_second * 1000]
            print(result['eeg']['signals'].shape)
            if not offline_transform is None:
                try:
                    for signal_type in signal_types:
                        if signal_type in offline_transform:
                            result[signal_type] = offline_transform[signal_type](result[signal_type])
                except (KeyError, ValueError) as e:
                    print(f'Error in processing record {file_name}: {e}')
                    return None
            # extract experimental signals
            start_at = 0
            if chunk_size <= 0:
                dynamic_chunk_size = result['eeg']['signals'].shape[1] - start_at
            else:
                dynamic_chunk_size = chunk_size

            # chunk with chunk size
            end_at = dynamic_chunk_size
            # calculate moving step
            step = dynamic_chunk_size - overlap
            num_channel = len(result['eeg']['channels'])
            while end_at <= result['eeg']['signals'].shape[1]:
                clip_sample = result['eeg']['signals'][:num_channel, start_at:end_at]

                t_eeg = clip_sample

                clip_id = f'{write_pointer}_{file_name}'
                write_pointer += 1

                record_info = {
                    'clip_id': clip_id,
                    'start_at': start_at,
                    'end_at': end_at,
                    'signal_type': 'eeg'
                }
                record_info.update(trial_meta_info)
                eeg = {
                    'signals': t_eeg,
                    'sampling_rate': result['eeg']['sampling_rate'],
                    'channels': result['eeg']['channels']
                }
                yield {
                    'eeg': eeg,
                    'key': clip_id, 
                    'info': record_info
                    }

                start_at = start_at + step
                end_at = start_at + dynamic_chunk_size

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