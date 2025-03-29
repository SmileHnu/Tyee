#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2024, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : bciciv2a_dataset.py
@Time    : 2024/12/19 19:40:20
@Desc    : 
"""
import os
import mne
import torch
import numpy as np
import scipy.io as scio
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict

class BCICIV2ADataset(BaseDataset):
    def __init__(self,
                 root_path: str = './BCICIV_2a_mat',
                 offset: int = 0,
                 chunk_size: int = 7 * 250,
                 overlap: int = 0,
                 num_channel: int = 22,
                 skip_trial_with_artifacts: bool = False,
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
                 verbose: bool = True):
        params = {
            'root_path': root_path,
            'offset': offset,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'num_channel': num_channel,
            'skip_trial_with_artifacts': skip_trial_with_artifacts,
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
            'verbose': verbose
        }
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def set_records(self, root_path: str = './BCICIV_2a_mat', **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = os.listdir(root_path)
        file_list = [
            os.path.join(root_path, file) for file in file_list
            if file.endswith('.mat')
        ]

        return file_list
    
    @staticmethod
    def read_record(record: str, **kwargs) -> Dict:
        a_data = scio.loadmat(record)['data']

        result = {
            'a_data': a_data,
        }
        return result
    
    @staticmethod
    def process_record(record: str,
                       result: Dict,
                       signal_types: list,
                       offset: int = 0,
                       chunk_size: int = 7 * 250,
                       overlap: int = 0,
                       num_channel: int = 22,
                       skip_trial_with_artifacts: bool = False,
                       before_trial: Union[None, Callable] = None,
                       offline_transform: Union[None, Callable] = None,
                       **kwargs):

        if chunk_size <= 0:
            dynamic_chunk_size = 7 * 250
        else:
            dynamic_chunk_size = int(chunk_size)

        # get file name without extension
        file_name = os.path.splitext(os.path.basename(record))[0]
        # the last letter of the file name is the session, the rest is the subject
        subject = file_name[:-1]
        session = file_name[-1]
        write_pointer = 0
        a_data = result['a_data'].copy()
        for run_id in range(0, a_data.size):
            # a_data: (1, 9) struct, 1-3: 25 channel EOG test (eyes open, eyes closed, movement), 4-9: 6 runs

            a_data1 = a_data[0, run_id]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_artifacts = a_data3[5]
            a_X = np.transpose(a_X)  # to channel number, data point number
            
            # for EOG test, a_trial is []
            for trial_id in range(0, a_trial.size):
                trial_meta_info = {
                    'subject_id': subject,
                    'session': f'{subject}_{session}',
                    'run_id': f'{subject}_{session}_{run_id}',
                    'trial_id': f'{subject}_{session}_{run_id}_{trial_id}',
                }

                if (a_artifacts[trial_id] != 0 and skip_trial_with_artifacts):
                    continue

                start_at = int(a_trial[trial_id] + offset)
                end_at = start_at + dynamic_chunk_size
                step = dynamic_chunk_size - overlap

                if trial_id < a_trial.size - 1:
                    trial_end_at = int(a_trial[trial_id + 1])
                else:
                    trial_end_at = a_X.shape[1]

                while end_at <= trial_end_at:
                    clip_id = f'{write_pointer}_{file_name}'

                    record_info = {
                        'signal_types': ['eeg'],
                        'start_at': start_at,
                        'end_at': end_at,
                        'clip_id': clip_id
                    }
                    record_info.update(trial_meta_info)

                    t_eeg = a_X[:num_channel, start_at:end_at]
                    result['eeg'] = {
                        'signals': t_eeg,
                        'sampling_rate': 250,
                        }
                    if not offline_transform is None:
                        try:
                            for signal_type in signal_types:
                                if signal_type in offline_transform:
                                    result[signal_type] = offline_transform[signal_type](result[signal_type])
                        except (KeyError, ValueError) as e:
                            print(f'Error in processing record {file_name}: {e}')
                            return None
                    
                    
                    record_info['label'] = int(a_y[trial_id])
                    eeg = {
                        'signals': result['eeg']['signals'] ,
                        'sampling_rate': result['eeg']['sampling_rate'],
                    }
                    yield {'eeg': eeg, 
                           'key': clip_id, 
                           'info': record_info}
                   

                    write_pointer += 1

                    start_at = start_at + step
                    end_at = start_at + dynamic_chunk_size
    
    def __getitem__(self, index):
        info = self.read_info(index)
        
        signal_index = str(info['clip_id'])
        signal_record = str(info['record_id'])

        result = {}
        for signal_type in self.signal_types:
            result[signal_type] = self.read_signal(signal_record, signal_index, signal_type)

        result['label'] = info['label']-1
        if self.label_transform is not None:
            result['label'] = self.label_transform(result['label'])
        if self.online_transform is not None:
            for signal_type in self.signal_types:
                if signal_type in self.online_transform:
                    result[signal_type] = self.online_transform[signal_type](result[signal_type])
                if 'ToIndexChannels' not in [transform.__class__.__name__ for transform in self.online_transform[signal_type].transforms]:
                    if 'channels' in result[signal_type]:
                        del result[signal_type]['channels']
        return result