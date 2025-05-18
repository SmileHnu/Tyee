#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : wesad_dataset.py
@Time    : 2025/03/30 15:42:48
@Desc    : 
"""
import os
import torch
import scipy
import numpy as np
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter
from typing import Callable, Union, Dict, List, Tuple

# from dataset.base_dataset import BaseDataset
from dataset.base_dataset import BaseDataset


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Helper function for butter_bandpass_filter
    Creates a scipy filter
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    :param lowcut: lower cut-off value for butterworth filter (Hz)
    :param highcut: upper cut-off value for butterworth filter (Hz)
    :param fs: frequency of signal in Hz
    :param order: order of filter
    :return:(ndarray, ndarray)  Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.scipy filter object
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Filters the signal with specified butter-bandpass filter
    :param data: single-channel signal (one-dimensional)
    :param lowcut: . of the filter
    :param highcut: . of the filter
    :param fs: input sampling frequency
    :param order: . of the filter
    :return: filtered signal of same shape
    """
    data = scipy.signal.detrend(data)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def process_window_spec_ppg(sig, freq, resolution, min_hz, max_hz):
    """
    Processes a window of PPG signal into spectral representation. Custom implementation of STFT.
    The channels are averaged, the signal is downsampled and the FFT is computed. Only amplitudes within relevant
    range are returned.
    :param sig: PPG signal of shape (n_samples, n_channels)
    :param freq: signal frequency (Hz)
    :param resolution: Number of points for the FFT algorithm
    :param min_hz: Lower cutoff frequency for spectrum
    :param max_hz: Higher cutoff frequency for spectrum
    :return: Spectrogram of shape (n_steps, n_freq_bins)
    """
    filt = lambda x: butter_bandpass_filter(x, 0.4, 4, fs=freq, order=4)
    sig = np.apply_along_axis(filt, 0, sig)
    sig = (sig - sig.mean(axis=0)[None, :]) / (sig.std(axis=0)[None, :] + 1e-10)
    sig = sig.mean(axis=-1)  # average over channels if multiple present

    sig = scipy.signal.resample(
        sig, int(len(sig) * 25 / freq)
    )  # resample to 25 hz (optional but faster)
    # do FFT
    if resolution > len(sig):
        sig = np.pad(sig, (0, resolution - len(sig)))
    y = np.abs(rfft(sig, axis=0))
    freq = rfftfreq(len(sig), 1 / 25)
    # extract relevant frequencies
    y = y[(freq > min_hz) & (freq < max_hz)]
    return y


def process_window_spec_acc(sig, freq, resolution: int, min_hz: float, max_hz: float):
    """
    Processes a window of accelerometer signal into spectral representation. Custom implementation of STFT.
    The signal is downsampled, FFT is computed and the channels are averaged. Only amplitudes within relevant
    range are returned.
    :param sig: ACC signal of shape (n_samples, n_channels)
    :param freq: signal frequency (Hz)
    :param resolution: Number of points for the FFT algorithm
    :param min_hz: Lower cutoff frequency for spectrum
    :param max_hz: Higher cutoff frequency for spectrum
    :return: Spectrogram of shape (n_steps, n_freq_bins)
    """
    filt = lambda x: butter_bandpass_filter(x, 0.4, 4, fs=freq, order=4)
    sig = np.apply_along_axis(filt, 0, sig)
    sig = (sig - sig.mean(axis=0)[None, :]) / (sig.std(axis=0)[None, :] + 1e-10)

    sig = scipy.signal.resample(
        sig, int(len(sig) * 25 / freq)
    )  # resample to 25 hz (optional but faster)
    # do FFT
    if resolution > len(sig):
        sig = np.pad(sig, ((0, resolution - len(sig)), (0, 0)))
    y = np.abs(rfft(sig, axis=0))
    freq = rfftfreq(len(sig), 1 / 25)
    # extract relevant frequencies
    y = y[(freq > min_hz) & (freq < max_hz), :]
    return y.mean(axis=-1)


def prepare_session_spec(ppg, acc, ppg_freq, acc_freq, win_size, stride, n_bins, min_hz, max_hz):
    """
    Prepares the spectral-domain input. Performs a custom version of STFT with channel averaging.
    Only returns frequencies within a relevant range.
    :param ppg: the ppg signal with shape (n_samples, n_ppg_channels)
    :param acc: the acceleromenter signal with shape (n_samples, n_acc_channels)
    :param ppg_freq: frequency of the PPG signal
    :param acc_freq: frequency of the ACC signal
    :param win_size: size in seconds of the signal window for fourier transform
    :param stride: size in seconds of the window stride for STFT
    :param n_bins: number of frequency bins. Can be either 64 or 256 for model compatibility
    :param min_hz: minimal relevant frequency in Hz.
    :param max_hz: maximal relevant frequency in Hz.
    :return: processed signal of shape (n_steps, n_bins, n_channels),
            where n_channels=2 stands for the aggregated PPG and ACC signals
    """
    fft_winsize = (
        535 if n_bins == 64 else (4 * 535 - 5)
    )
    ppgs = []
    ppg_wsize = win_size * ppg_freq
    for i in range(0, len(ppg) - ppg_wsize + 1, ppg_freq * stride):
        ppgs.append(
            process_window_spec_ppg(
                ppg[i : i + ppg_wsize], ppg_freq, fft_winsize, min_hz, max_hz
            )
        )

    accs = []
    acc_wsize = win_size * acc_freq
    for i in range(0, len(acc) - acc_wsize + 1, acc_freq * stride):
        accs.append(
            process_window_spec_acc(
                acc[i : i + acc_wsize], acc_freq, fft_winsize, min_hz, max_hz
            )
        )

    sig = np.stack([ppgs, accs], axis=-1)

    # normalize
    sig = (sig - sig.mean()) / (sig.std() + 1e-10)

    assert not np.isnan(sig).any()
    return sig.astype(np.float32)


def process_window_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut):
    """
    Prepares time-domain PPG signal as input. Preprocessing consists of filtering, averaging & resampling.
    :param ppg: PPG signal of shape (n_samples, n_channels)
    :param ppg_freq: frequency of signal
    :param target_freq: desired frequency for input
    :param filter_lowcut: lower cutoff frequency for bandpass filtering
    :param filter_highcut: upper cutoff frequency for bandpass filtering
    :return: time-domain signal of shape (n_samples_new,)
    """
    filt = lambda x: butter_bandpass_filter(x, filter_lowcut, filter_highcut, ppg_freq)
    ppg = np.apply_along_axis(filt, 0, ppg)
    ppg = ppg.mean(axis=-1)  # average over channels if multiple present
    ppg = scipy.signal.resample_poly(ppg, target_freq, ppg_freq)
    ppg = np.expand_dims(ppg, -1)
    return ppg


def prepare_session_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut):
    """
    Prepares the time-domain input. Preprocessing mainly consists of filtering, resampling and standardizing.
    :param ppg: the ppg signal with shape (n_samples, n_channels)
    :param ppg_freq: frequency of the signal
    :param target_freq: desired input frequency for model
    :param filter_lowcut: lower cutoff frequency for bandpass filter
    :param filter_highcut: upper cutoff frequency for bandpass filter
    :return: processed signal of shape (n_samples_new,)
    """
    # only feed ppg signals as time-domain features
    sig = process_window_time(ppg, ppg_freq, target_freq, filter_lowcut, filter_highcut)

    # normalize
    sig = (sig - sig.mean()) / (sig.std() + 1e-10)

    assert not np.isnan(sig).any()
    return sig.astype(np.float32)


def prepare_session_labels(hr, n_frames):
    """
    Add offset to session labels to compensate for the length of the feature window.
    :param hr: labels of shape (n_steps, )
    :param n_frames: number of steps per feature window
    :return: labels of shape (n_steps - n_frames + 1, )
    """
    offset = n_frames - 1
    assert not np.isnan(hr).any()
    return hr[offset:].astype(np.float32)


def get_strided_windows(ds, win_size, stride):
    """
    Generates a sliding window view of a np.ndarray along the first axis
    :param win_size: number of samples in window
    :param stride: number of skipped samples between consecutive windows
    :return: tf.data.Dataset yielding tensors of shape (win_size, ) + old_shape
    """
    res = []
    num_samples = (len(ds) - win_size) // stride + 1
    for idx in range(0, num_samples):
        start = idx * stride
        end = start + win_size
        if end > len(ds):
            break
        res.append(ds[start : end, ...])
    return res


class DaLiADataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './lingyus/erp-based-brain-computer-interface-recordings-1.0.0',
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
        super().__init__(**params)
    
    def set_records(self, root_path, **kwargs):        
        records = list(
            # glob.glob(
            #     os.path.join(*[root_path, "DaLia/PPG_FieldStudy", "S*", "S*.pkl"])
            # )
            Path(root_path).rglob("S*.pkl")
        )
        records = sorted(records)
        return records
    
    def read_record(
        self,
        record: tuple | str,
        **kwargs
    ) -> Dict:
        fname = record

        ppg_freq = 64.0
        acc_freq = 32.0

        ds = ds = np.load(fname, allow_pickle=True, encoding="bytes")
        name = os.path.split(fname)[1][:-4]
        acc_data = ds[b"signal"][b"wrist"][b"ACC"]
        ppg_data = ds[b"signal"][b"wrist"][b"BVP"]
        hr_data = ds[b"label"]
        ppg = {
            "data": ppg_data.T,
            "freq": ppg_freq
        }
        acc = {
            "data": acc_data.T,
            "freq": acc_freq
        }
        hr = {
            "data": hr_data
        }
        segments = []
        segments.append({
            "start": 0,
            "end": len(ppg_data) / ppg_freq,
            "value":{
                'hr': hr
            }
        })
        return {
            'signals':{
                'ppg': ppg,
                'acc': acc
            },
            'labels':{
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
        
    def get_subject_id(self, file_name) -> str:
        # Extract the subject ID from the file name
        # Assuming the file name format is like "subject_id_record_id"
        # You can modify this logic based on your actual file naming convention
        return int(file_name[1:])
    
    def get_segment_id(self, file_name, idx) -> str:
        # Extract the segment ID from the file name
        # Assuming the segment ID is the same as the file name in this case
        return f'{idx}_{file_name}'
    
    def get_trial_id(self, idx) -> str:
        # Extract the trial ID from the index
        # Assuming the trial ID is the same as the index in this case
        return str(idx)
    

