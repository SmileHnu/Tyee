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
from typing import Any, Callable, Union, Dict, Generator, List

import mne
import scipy.interpolate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
L_FREQ, H_FREQ = 40, 300 # Lower and upper filtration bounds
CHANNELS_NUM = 62        # Number of channels in ECoG data
WAVELET_NUM = 40         # Number of wavelets in the indicated frequency range, with which the convolution is performed
DOWNSAMPLE_FS = 100      # Desired sampling rate
time_delay_secs = 0.2    # Time delay hyperparameter

current_fs = DOWNSAMPLE_FS

def reshape_column_ecog_data(multichannel_signal: np.ndarray):
    return multichannel_signal.T # (time, features) -> (features, time)

def filter_ecog_data(multichannel_signal: np.ndarray, fs=1000, powerline_freq=50):
    """
    Harmonics removal and frequency filtering
    :param multichannel_signal: Initial multi-channel signal
    :param fs: Sampling rate
    :param powerline_freq: Grid frequency
    :return: Filtered signal
    """
    harmonics = np.array([i * powerline_freq for i in range(1, (fs // 2) // powerline_freq)])

    print("Starting...")
    signal_filtered = mne.filter.filter_data(multichannel_signal,
                                             fs, l_freq=L_FREQ, h_freq=H_FREQ)  # remove all frequencies between l and h
    print("Noise frequencies removed...")
    signal_removed_powerline_noise = mne.filter.notch_filter(signal_filtered,
                                                             fs, freqs=harmonics)  # remove powerline  noise
    print("Powerline noise removed...")
    
    return signal_removed_powerline_noise

def normalize(multichannel_signal: np.ndarray, return_values = None):
    """
    standardization and removal of the median  from each channel
    :param multichannel_signal: Multi-channel signal
    :param return_values: Whether to return standardization parameters. By default - no
    """
    print("Normalizing...")
    means = np.mean(multichannel_signal, axis=1, keepdims=True)
    stds = np.std(multichannel_signal, axis=1, keepdims=True)
    transformed_data = (multichannel_signal - means) / stds
    common_average = np.median(transformed_data, axis=0, keepdims=True)
    transformed_data = transformed_data - common_average
    if return_values:
        return transformed_data, (means, stds)
    print("Normalized...")
    return transformed_data

def compute_spectrogramms(multichannel_signal : np.ndarray, fs=1000, freqs=np.logspace(np.log10(L_FREQ), np.log10(H_FREQ), WAVELET_NUM),
                          output_type='power'):
    """
    Compute spectrogramms using wavelet transforms

    :param freqs: wavelet frequencies to uses
    :param fs: Sampling rate
    :return: Signal spectogramms in shape (channels, wavelets, time)
    """
    
    num_of_channels = len(multichannel_signal)

    print("Computing wavelets...")
    spectrogramms = mne.time_frequency.tfr_array_morlet(multichannel_signal.reshape(1, num_of_channels, -1), sfreq=fs,
                                                        freqs=freqs, output=output_type, verbose=10, n_jobs=6)[0]
    
    
    print("Wavelet spectrogramm computed...")
    
    return spectrogramms


def downsample_spectrogramms(spectrogramms: np.ndarray, cur_fs=1000, needed_hz=H_FREQ, new_fs = None):
    """
    Reducing the sampling rate of spectrograms
    :param spectrogramms: Original set of spectrograms
    :param cur_fs: Current sampling rate
    :param needed_hz: The maximum frequency that must be unambiguously preserved during compression
    :param new_fs: The required sampling rate (interchangeable with needed_hz)
    :return: Decimated signal
    """
    print("Downsampling spectrogramm...")
    if new_fs == None:
        new_fs = needed_hz * 2    
    downsampling_coef = cur_fs // new_fs
    assert downsampling_coef > 1
    downsampled_spectrogramm = spectrogramms[:, :, ::downsampling_coef]
    print("Spectrogramm downsampled...")
    return downsampled_spectrogramm


def normalize_spectrogramms_to_db(spectrogramms: np.ndarray, convert = False):
    """
    Optional conversion to db, not used in the final version
    """
    if convert:
        return np.log10(spectrogramms+1e-12)
    else:
        return spectrogramms


def interpolate_fingerflex(finger_flex, cur_fs=1000, true_fs=25, needed_hz=DOWNSAMPLE_FS, interp_type='cubic'):
    
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
    interpolated_finger_flex_funcs = [scipy.interpolate.interp1d(ts, finger_flex_true_fs_ch, kind=interp_type) for
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
    spectrogramms_cropped = spectrogramms[..., :spectrogramms.shape[2]-time_delay]
    return finger_flex_cropped, spectrogramms_cropped


class BCICIV4Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV4',
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        before_segment_transform: Union[None, List[Callable]] = None,
        offline_signal_transform: Union[None, List[Callable]] = None,
        offline_label_transform: Union[None, List[Callable]] = None,
        online_signal_transform: Union[None, List[Callable]] = None,
        online_label_transform: Union[None, List[Callable]] = None,
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
            'start_offset': start_offset,
            'end_offset': end_offset,
            'before_segment_transform': before_segment_transform,
            'offline_signal_transform': offline_signal_transform,
            'offline_label_transform': offline_label_transform,
            'online_signal_transform': online_signal_transform,
            'online_label_transform': online_label_transform,
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

    def read_record(self, record: str, **kwargs):
        
        data = scio.loadmat(record)
        test_label = scio.loadmat(record.replace('comp.mat','testlabels.mat'))
        train_data = data['train_data'].astype(np.float64).T
        train_dg = {
            'data': data['train_dg'].astype(np.float64).T,
            'freq': 25
        }
        test_data = data['test_data'].astype(np.float64).T
        test_dg = {
            'data': test_label['test_dg'].astype(np.float64).T,
            'freq': 25
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

        for idx, segment in enumerate(self.segment_split(signals, labels)):
            seg_signals = segment['signals']
            seg_labels = segment['labels']
            seg_info = segment['info']
            # print(signals['eeg']['data'].shape)
            # print(label['label']['data'])
            segment_id = self.get_segment_id(meta['file_name'], idx)
            seg_labels['dg']['data'] = interpolate_fingerflex(finger_flex= seg_labels['dg']['data'])

            seg_signals['ecog']['data'] = normalize_spectrogramms_to_db(spectrogramms=
                            downsample_spectrogramms(spectrogramms=
                            compute_spectrogramms(multichannel_signal=
                            filter_ecog_data(multichannel_signal=
                            normalize(multichannel_signal=
                            seg_signals['ecog']['data']))), new_fs = DOWNSAMPLE_FS))
            seg_labels['dg']['data'], seg_signals['ecog']['data'] = crop_for_time_delay(seg_labels['dg']['data'],
                                                                                seg_signals['ecog']['data'], time_delay_secs,
                                                                                current_fs)
            scaler = MinMaxScaler()
            scaler.fit(seg_labels['dg']['data'].T)
            seg_labels['dg']['data'] = scaler.transform(seg_labels['dg']['data'].T).T

            transformer = RobustScaler(unit_variance=True, quantile_range=(0.1, 0.9))
            transformer.fit(seg_signals['ecog']['data'].T.reshape(-1,WAVELET_NUM*CHANNELS_NUM))
            seg_signals['ecog']['data'] = transformer.transform(seg_signals['ecog']['data'].T.reshape(-1,WAVELET_NUM*CHANNELS_NUM)).reshape(-1,\
                                                                                            WAVELET_NUM, CHANNELS_NUM).T
            seg_signals = self.apply_transform(self.offline_signal_transform, seg_signals)
            seg_labels = self.apply_transform(self.offline_label_transform, seg_labels)

            
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
        # print(sample_id)
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
    