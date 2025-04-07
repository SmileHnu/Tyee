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
import copy
import scipy.io as scio
from scipy.ndimage import label
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d
import numpy as np
from dataset import BaseDataset
from typing import Any, Callable, Union, Dict, Generator

class BCICIV4Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV4',
        chunk_size: int = 40,
        overlap: int = 0,
        time_delay_secs: float = 0.2,
        num_channel: int = 62,
        signal_types: list = ['ecog'],
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
        verbose: bool = True,
    ) -> None:
        # if io_path is None:
        #     io_path = get_random_dir_path(dir_prefix='datasets')

        # pass all arguments to super class
        params = {
            'root_path': root_path,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'time_delay_secs': time_delay_secs,
            'num_channel': num_channel,
            'signal_types': signal_types,
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

    @staticmethod
    def read_record(record: str, **kwargs):
        
        data = scio.loadmat(record)
        test_label = scio.loadmat(record.replace('comp.mat','testlabels.mat'))
        train_data = data['train_data'].astype(np.float64).T
        train_dg = data['train_dg'].astype(np.float64).T
        test_data = data['test_data'].astype(np.float64).T
        test_dg = test_label['test_dg'].astype(np.float64).T
        result = {
            'train':{
                'data': train_data,
                'dg': train_dg
            },
            'test':{
                'data': test_data,
                'dg': test_dg
            }
        }
        return result
        
    @staticmethod
    def process_record(
        record, 
        result,
        signal_types,
        chunk_size,
        overlap,
        time_delay_secs,
        offline_transform,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        file_name = os.path.splitext(os.path.basename(record))[0]
        subject_id = file_name.split('_')[0]
        raw_result = copy.deepcopy(result)
        for key in ['train', 'test']:
            # print(f'Processing {file_name}...')
            result = {
                'ecog' : {
                    'signals': raw_result[key]['data'].copy(),
                    'sampling_rate': 1000,
                    'channels': [f"{i}" for i in range(raw_result[key]['data'].shape[0])]
                }
            }
            if not offline_transform is None:
                try:
                    for signal_type in signal_types:
                        if signal_type in offline_transform:
                            result[signal_type] = offline_transform[signal_type](result[signal_type])
                except (KeyError, ValueError) as e:
                    print(f'Error in processing record {file_name}: {e}')
                    return None
                
            new_freq = result['ecog']['sampling_rate']
            # print(raw_result[key]['dg'].shape)
            raw_result[key]['dg'] = interpolate_fingerflex(raw_result[key]['dg'], needed_hz=new_freq,)
            dg = raw_result[key]['dg'].copy()
            ecog = result['ecog']['signals'].copy()
            dg, ecog = crop_for_time_delay(dg, ecog, time_delay_sec=time_delay_secs , fs=new_freq)
            scaler = MinMaxScaler()
            scaler.fit(dg.T)
            dg = scaler.transform(dg.T).T
            
            start = 0
            end = chunk_size
            # 计算重叠步长
            stride = chunk_size - overlap
            idx = 0
            while end <= dg.shape[-1]:
                # print(f'Processing trial: {idx}')
                clip_id = f'{idx}_{key}_{file_name}'
                info = {
                    'clip_id': clip_id,
                    'subject_id': subject_id,
                    'trial_id': key,
                    'start_at': start,
                    'end_at': end,
                    'label': dg[...,start:end]
                }
                result = {
                    'ecog': {
                        'signals': ecog[..., start:end],
                        'sampling_rate': result['ecog']['sampling_rate'],
                        'channels': result['ecog']['channels'],
                    }
                }

                result.update({
                    'key': clip_id,
                    'info': info
                })
                yield result
                start = start + stride
                end = start + chunk_size
                idx += 1


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
            for signal_type in self.signal_types:
                if 'channels' in result[signal_type]:
                    del result[signal_type]['channels']
        return result
    
def interpolate_fingerflex(finger_flex, cur_fs=1000, true_fs=25, needed_hz=100, interp_type='cubic'):
    
    """
    Interpolation of the finger motion recording to match the new sampling rate
    :param finger_flex: Initial sequences with finger flexions data
    :param cur_fs: ECoG sampling rate
    :param true_fs: Actual finger motions recording sampling rate
    :param needed_hz: Required sampling rate
    :param interp_type: Type of interpolation. By default - cubic
    :return: Returns an interpolated set of finger motions with the desired sampling rate
    """
    
    print("Interpolating fingerflex...")
    downscaling_ratio = cur_fs // true_fs
    print("Computing true_fs values...")
    finger_flex_true_fs = finger_flex[:, ::downscaling_ratio]
    finger_flex_true_fs = np.c_[finger_flex_true_fs,
        finger_flex_true_fs.T[-1]]  # Add as the last value on the interpolation edge the last recorded
    # Because otherwise it is not clear how to interpolate the tail at the end

    upscaling_ratio = needed_hz // true_fs
    
    ts = np.asarray(range(finger_flex_true_fs.shape[1])) * upscaling_ratio
    
    print("Making funcs...")
    interpolated_finger_flex_funcs = [interp1d(ts, finger_flex_true_fs_ch, kind=interp_type) for
                                     finger_flex_true_fs_ch in finger_flex_true_fs]
    ts_needed_hz = np.asarray(range(finger_flex_true_fs.shape[1] * upscaling_ratio)[
                              :-upscaling_ratio])  # Removing the extra added edge
    
    print("Interpolating with needed frequency")
    interpolated_finger_flex = np.array([[interpolated_finger_flex_func(t) for t in ts_needed_hz] for
                                         interpolated_finger_flex_func in interpolated_finger_flex_funcs])
    return interpolated_finger_flex


def crop_for_time_delay(finger_flex : np.ndarray, spectrogramms : np.ndarray, time_delay_sec : float, fs : int):
    """
    Taking into account the delay between brain waves and movements
    :param finger_flex: Finger flexions
    :param spectrogramms: Computed spectrogramms
    :param time_delay_sec: time delay hyperparameter
    :param fs: Sampling rate
    :return: Shifted series with a delay
    """

    time_delay = int(time_delay_sec*fs)

    # the first motions do not depend on available data
    finger_flex_cropped = finger_flex[..., time_delay:] 
    # The latter spectrograms have no corresponding data
    spectrogramms_cropped = spectrogramms[..., :spectrogramms.shape[-1]-time_delay]
    return finger_flex_cropped, spectrogramms_cropped