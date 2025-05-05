#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : tuev_dataset.py
@Time    : 2025/02/26 16:03:52
@Desc    : 
"""

import os
import mne
import torch
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator
from dataset.constants.standard_channels import EEG_CHANNELS_ORDER

class TUEVDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './tuh_eeg_events/v2.0.1/edf/train',
        start_offset: float = -2.0,
        end_offset: float = 2.0,
        signal_types: list = ['eeg'],
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        after_trial: Union[Callable, None] = None,
        after_session: Union[Callable, None] = None,
        after_subject: Union[Callable, None] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_mode: str = 'lmdb',
        num_worker: int = 0,
        verbose: bool = True,
    ) -> None:
        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'signal_types': signal_types,
            'before_segment_transform': before_segment_transform,
            'offline_signal_transform': offline_signal_transform,
            'offline_label_transform': offline_label_transform,
            'online_signal_transform': online_signal_transform,
            'online_label_transform': online_label_transform,
            'after_trial': after_trial,
            'after_session': after_session,
            'after_subject': after_subject,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }
        self.__dict__.update(params)
        super().__init__(**params)
        # save all arguments to __dict__
    
    def set_records(self, root_path: str = None, **kwargs):
        assert os.path.exists(
            root_path
        ), f'root_path ({root_path}) does not exist. Please download the dataset and set the root_path to the downloaded path.'

        file_list = []
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                if file.endswith('.edf'):
                    file_list.append(os.path.join(dirpath, file))
        
        file_list = sorted(file_list)
        return file_list

    @staticmethod
    def read_record(record: str, **kwargs):
        # print(f'Processing record: {record}')
        Rawdata = mne.io.read_raw_edf(record, preload=True)
        channel_names = Rawdata.ch_names
        new_channel_names = [name.split(' ')[-1].split('-')[0] for name in channel_names]

        mapping = {old_name: new_name for old_name, new_name in zip(channel_names, new_channel_names)}
        Rawdata.rename_channels(mapping)

        drop_channels = [ch for ch in Rawdata.ch_names if ch not in EEG_CHANNELS_ORDER]
        Rawdata.drop_channels(drop_channels)
        channel_names = Rawdata.ch_names
        info = Rawdata.info
        freq = info['sfreq']
        eeg_channels = info['ch_names']
        _, times = Rawdata[:]
        data = Rawdata.get_data(units='uV')
        print(f"Data shape: {data.shape}")
        RecFile = record[0:-3] + "rec"
        eventData = np.genfromtxt(RecFile, delimiter=",")
        Rawdata.close()
        eeg = {
            'data': data,
            'freq': freq,
            'channels': eeg_channels,
        }
        [numEvents, z] = eventData.shape
        [numChan, numPoints] = data.shape
        segments = []
        for i in range(numEvents):
            chan = int(eventData[i, 0])
            start = (np.where((times) >= eventData[i, 1])[0][0]) /freq
            end = (np.where((times) >= eventData[i, 2])[0][0]) /freq
            # print(start, end)
            label = int(eventData[i, 3])
            segments.append({
                'start': start,
                'end': end,
                'value':{
                    'event':{
                        'data': label,
                    },
                }
            })
        return {
            'signals':{
                'eeg': eeg
            },
            'labels':{
                'segments': segments,
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
        try:
            signals = self.apply_transform(self.before_segment_transform, signals)
        except (KeyError, ValueError) as e:
            print(f'Error in processing record {meta["file_name"]}: {e}')
            return None

        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_label = segment['labels']
            seg_info = segment['info']
            # print(signals['eeg']['data'].shape)
            # print(label['label']['data'])
            segment_id = self.get_segment_id(meta['file_name'], idx)
            try:
                seg_signals = self.apply_transform(self.offline_signal_transform, seg_signals)
                seg_label = self.apply_transform(self.offline_label_transform, seg_label)
            except (KeyError, ValueError) as e:
                print(f'Error in processing record {meta["file_name"]}: {e}')
                return None
            
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

    def __getitem__(self, index: int) -> Dict:
        """
        Retrieve a dataset item by index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing signal data and labels.
        """
        info = self.read_info(index)
        sample_id = str(info['sample_id'])
        record = str(info['record_id'])
        # print(record, sample_id)
        signals = self.read_signal(record, sample_id)
        signals = self.apply_transform(self.online_signal_transform, signals)

        labels = self.read_label(record, sample_id)
        labels = self.apply_transform(self.online_label_transform, labels)

        return self.assemble_sample(signals,labels)

    def get_subject_id(self, file_name) -> str:
        # Extract the subject ID from the file name
        # Assuming the file name format is like "subject_id_record_id.edf"
        # You can modify this logic based on your actual file naming convention
        return file_name.split('_')[0]
    
    def get_segment_id(self, file_name, idx) -> str:
        # Extract the segment ID from the file name
        # Assuming the segment ID is the same as the file name in this case
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        # Extract the trial ID from the index
        # Assuming the trial ID is the same as the index in this case
        return str(idx)
    
    def get_sample_ids(self, segment_id, sample_len) -> str:
        # Extract the sample ID from the file name and index
        # Assuming the sample ID is a combination of the file name and index
        return [f"{i}_{segment_id}" for i in range(sample_len)]