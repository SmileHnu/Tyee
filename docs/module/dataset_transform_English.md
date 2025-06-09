# tyee.dataset.transform

The `tyee.dataset.transform` module provides a rich set of signal preprocessing methods to help users extract features and construct signal representations.

| Class Name                               | Function/Purpose                                             |
| ---------------------------------------- | ------------------------------------------------------------ |
| [`BaseTransform`](#basetransform)       | Base class for all transform classes, defining the basic interface and usage for transform operations. |
| [`CommonAverageRef`](#commonaverageref) | Performs Common Average Referencing (CAR) or Common Median Referencing (CMR) baseline correction on signal data for each sample point. |
| [`PickChannels`](#pickchannels)         | Selects specified channels from the signal data.             |
| [`OrderChannels`](#orderchannels)       | Reorders signal channels according to a specified order, with optional padding for missing channels. |
| [`ToIndexChannels`](#toindexchannels)   | Converts channel names in the signal data to their corresponding indices based on a provided master list. |
| [`UniToBiTransform`](#unitobitransform) | Converts unipolar signals to bipolar signals by calculating the difference between specified pairs of channels. |
| [`Crop`](#crop)                         | General-purpose transform to crop a specified number of points from the left (start) and/or right (end) of a signal along a given axis. |
| [`Filter`](#filter)                     | Applies a filter (e.g., band-pass, low-pass, high-pass) to the signal data, based on `mne.filter.filter_data`. |
| [`NotchFilter`](#notchfilter)           | Applies a notch filter to remove specific frequency components (e.g., power line noise) from signal data, based on `mne.filter.notch_filter`. |
| [`Cheby2Filter`](#cheby2filter)         | Applies a Chebyshev type II digital or analog filter to the signal data, based on `scipy.signal.cheby2`. |
| [`Detrend`](#detrend)                   | Removes the linear trend from signal data along a specified axis, based on `scipy.signal.detrend`. |
| [`Compose`](#compose)                   | Composes a sequence of transforms to be applied sequentially to the input data. |
| [`ForEach`](#foreach)                   | Applies a sequence of specified sub-transforms to each segment along the first dimension of the data (e.g., epochs or trials). |
| [`Lambda`](#lambda)                     | Applies a user-defined lambda function or any callable to the 'data' field in the signal dictionary for custom operations. |
| [`Mapping`](#mapping)                   | Applies a dictionary mapping to the values in the 'data' field. Handles scalars directly and maps elements individually for arrays/iterables. |
| [`Concat`](#concat)                     | Concatenates 'data' arrays from multiple input signal dictionaries along a specified axis, attempts to merge channel names and retain frequency. |
| [`Stack`](#stack)                       | Stacks 'data' arrays from multiple input signal dictionaries along a new axis. |
| [`ZScoreNormalize`](#zscorenormalize)   | Performs Z-score normalization ((x - mean) / std) on signal data. |
| [`MinMaxNormalize`](#minmaxnormalize)   | Performs min-max normalization on signal data, scaling it to a specified range (typically 0 to 1). |
| [`QuantileNormalize`](#quantilenormalize) | Performs quantile normalization on signal data by dividing data by a specified quantile of its absolute values. |
| [`RobustNormalize`](#robustnormalize)   | Performs robust normalization using `(x - median) / IQR`, resistant to outliers. |
| [`Baseline`](#baseline)                 | Performs baseline correction on the signal by subtracting the mean of a specified interval. |
| [`Mean`](#mean)                         | Calculates the mean of the 'data' field in a signal dictionary along a specified axis, replacing the original data with the computed mean. |
| [`OneHotEncode`](#onehotencode)         | Converts input data (typically categorical labels) to one-hot encoded format. |
| [`Pad`](#pad)                           | Pads the 'data' array in a signal dictionary along a specified axis. |
| [`Resample`](#resample)                 | Resamples signal data to a specified target frequency using MNE library functions. |
| [`Downsample`](#downsample)             | Downsamples signal data from the current frequency to a specified target frequency by selecting every Nth sample. |
| [`Interpolate`](#interpolate)           | Upsamples signal data from the current frequency to a specified target frequency using interpolation (e.g., linear, cubic). |
| [`Reshape`](#reshape)                   | Reshapes the 'data' array in a signal dictionary to a specified target shape. |
| [`Transpose`](#transpose)               | Transposes the axes of the 'data' array in a signal dictionary. |
| [`Squeeze`](#squeeze)                   | Removes single-dimensional entries from the shape of the 'data' array in a signal dictionary. |
| [`ExpandDims`](#expanddims)             | Inserts a new axis (dimension) at a specified position in the 'data' array of a signal dictionary. |
| [`Insert`](#insert)                     | Inserts a specified value into the 'data' array of a signal dictionary at given indices along a specified axis. |
| [`ImageResize`](#imageresize)           | Resizes image data (typically in the 'data' field) to a specified size using `torchvision.transforms.Resize`. |
| [`Scale`](#scale)                       | Applies numerical scaling to signal data by multiplying it with a specified scale factor. |
| [`Offset`](#offset)                     | Applies a numerical offset to signal data by adding a specified offset value. |
| [`Round`](#round)                       | Rounds the numerical values in signal data to the nearest integer. |
| [`Log`](#log)                           | Applies a natural logarithm transformation to signal data, typically adding a small epsilon to data to avoid log(0). |
| [`Select`](#select)                     | Selects a subset of entries from an input dictionary based on a specified key or list of keys. |
| [`SlideWindow`](#slidewindow)           | Calculates the start and end indices for sliding windows along a specified axis of the signal data. |
| [`WindowExtract`](#windowextract)       | Extracts data segments (windows) from the signal data based on window indices and stacks them. |
| [`CWTSpectrum`](#cwtspectrum)           | Computes the time-frequency representation of signal data using Continuous Wavelet Transform (CWT) with Morlet wavelets. |
| [`DWTSpectrum`](#dwtspectrum)           | Computes the Discrete Wavelet Transform (DWT) coefficients for signal data. |
| [`FFTSpectrum`](#fftspectrum)           | Computes the Fast Fourier Transform (FFT) magnitude spectrum of signal data. |
| [`ToImage`](#toimage)                   | Converts input signal data (typically 2D) into an image representation, involving normalization, colormapping, resizing, etc. |
| [`ToTensor`](#totensor)                 | Converts 'data' from a NumPy array format to a PyTorch tensor. |
| [`ToTensorFloat32`](#totensorfloat32)   | Converts 'data' to a PyTorch tensor and ensures its data type is float32. |
| [`ToTensorFloat16`](#totensorfloat16)   | Converts 'data' to a PyTorch tensor and ensures its data type is float16 (half). |
| [`ToTensorInt64`](#totensorint64)       | Converts 'data' to a PyTorch tensor and ensures its data type is int64 (long). |
| [`ToNumpy`](#tonumpy)                   | Converts 'data' from a PyTorch tensor format to a NumPy array. |
| [`ToNumpyFloat64`](#tonumpyfloat64)     | Converts 'data' to a NumPy array and ensures its data type is float64. |
| [`ToNumpyFloat32`](#tonumpyfloat32)     | Converts 'data' to a NumPy array and ensures its data type is float32. |
| [`ToNumpyFloat16`](#tonumpyfloat16)     | Converts 'data' to a NumPy array and ensures its data type is float16. |
| [`ToNumpyInt64`](#tonumpyint64)         | Converts 'data' to a NumPy array and ensures its data type is int64. |
| [`ToNumpyInt32`](#tonumpyint32)         | Converts 'data' to a NumPy array and ensures its data type is int32. |

## BaseTransform

We provide the `BaseTransform` base class, which defines the interface and usage for transform operations.

~~~python
class BaseTransform(source:str| list[str] = None, target: str = None)
~~~

**Parameters**

- `source (str | list[str])`: Specifies the signal type field or list of fields that this operation targets.
- `target (str)`: Specifies the field where the processed result is saved.

**Methods**

~~~python
transform(result:dict)->dict
~~~

Parameters:

- `result (dict)`: A dictionary containing the signal fields to be processed. Each field may include the following information:
  - `data`: Signal data.
  - `freq`: Sampling rate.
  - `channels`: Channel list.
  - `info`: Other descriptive information about the signal, such as lists of start and end indices for each sample, etc..

We have implemented some preprocessing operations based on `BaseTransform`. Additionally, users can customize new preprocessing operations based on `BaseTransform`.

[`Back to Top`](#tyeedatasettransform)

## CommonAverageRef

~~~python
class CommonAverageRef(axis:int=0, mode:str = 'median', source:str| list[str] = None, target: str = None)
~~~

Performs baseline referencing for each sampling point of the signal.

**Parameters**

- `axis (int)`: Determines which dimension to operate on, usually the channel dimension.
- `mode (str)`: Reference value selection. Provides 'median' and 'mean'. Extensible.

**Usage Example**

~~~python
results = {
    'eeg': {
        'data': np.random.randn(32,128),
        'freq': 200.0,
        'channels': [i for i in range(0,32)]
    }
}
t= CommonAverageRef(axis=0, mode='median', source='eeg', target='eeg')
results = t(results)
~~~

[`Back to Top`](#tyeedatasettransform)

## PickChannels

~~~python
class PickChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], source: str = None, target: str = None):
~~~

Selects specified channels from an input signal. If a channel specified for picking is not found in the input signal, a `KeyError` is raised.

**Parameters**

-   `channels (Union[str, List[str]])`: A list of channel names to select. Alternatively, this can be a string key that is used to dynamically import a list of channel names (e.g., from `dataset.constants`).
-   `source (str, optional)`: The key in the input dictionary that holds the signal data to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key under which the transformed signal data will be stored in the output dictionary. If `None` or the same as `source`, the original signal data is overwritten. Defaults to `None`.

**Usage Example**

~~~python
# Assume 'results' is a dict like:
# results = {
#     'eeg': {
#         'data': np.random.randn(4, 128), # Example: 4 channels
#         'channels': ['Fz', 'Cz', 'Pz', 'Oz'],
#         'freq': 200.0
#     }
# }
# Define a list of channels to pick (ensure these exist in results['eeg']['channels'])
# For example, if results['eeg']['channels'] are ['Fz', 'Cz', 'Pz', 'Oz']
# and we want to pick ['Fz', 'Pz']
pick_these_channels = ['Fz', 'Pz'] # Ensure these are valid channels from the input
# Initialize PickChannels to operate on 'eeg' and store in 'eeg_picked'
t = PickChannels(channels=pick_these_channels, source='eeg', target='eeg_picked')
# Apply the transform
processed_results = t(results)
# processed_results['eeg_picked']['data'] will have shape (2, 128)
# processed_results['eeg_picked']['channels'] will be ['Fz', 'Pz']
~~~

---
[`Back to Top`](#tyeedatasettransform)

## OrderChannels

~~~python
class OrderChannels(BaseTransform):
    def __init__(self, order: Union[str, List[str]], padding_value: float = 0, source: str = None, target: str = None):
~~~

Reorders the channels of an input signal according to a specified `order`. If channels in the desired `order` are not present in the input signal, a `KeyError` is raised. Channels present in the input signal but not in the `order` list are discarded. This transform is primarily used to standardize channel orders across different datasets.

**Parameters**

-   `order (Union[str, List[str]])`: A list specifying the desired order of channel names. This can also be a string key for dynamically importing a channel order list.
-   `padding_value (float, optional)`: Reserved parameter (currently unused in the implementation). Defaults to `0`.
-   `source (str, optional)`: The key in the input dictionary for the signal to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary for the transformed signal. Defaults to `None`.


**Usage Example**

~~~python
# Assume 'results' is a dict like:
# results = {
#     'eeg': {
#         'data': np.random.randn(3, 100), # Data for Cz, Fz, Pz
#         'channels': ['Cz', 'Fz', 'Pz'],   # Original order
#         'freq': 100.0
#     }
# }
# Define the desired channel order (reorder existing channels)
desired_order = ['Fz', 'Cz', 'Pz'] # Reorder: Fz and Cz swap positions, Pz remains at the end
# Initialize OrderChannels
t = OrderChannels(order=desired_order, padding_value=np.nan, source='eeg', target='eeg_ordered')
# Apply the transform
processed_results = t(results)
# processed_results['eeg_ordered']['data'] will have shape (3, 100)
# processed_results['eeg_ordered']['channels'] will be ['Fz', 'Cz', 'Pz']
# Data rows will be rearranged according to the new channel order:
# Row 0: original Fz data (was row 1)
# Row 1: original Cz data (was row 0)
# Row 2: original Pz data (remains row 2)
~~~

---
[`Back to Top`](#tyeedatasettransform)

## ToIndexChannels

~~~python
class ToIndexChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], strict_mode: bool = False, source: str = None, target: str = None):
~~~

Converts channel names in the input signal's `channels` list to their corresponding integer indices based on a provided master list of `channels`. The `data` array itself is not modified by this transform as per the provided Python code; only the `channels` field is updated.

**Parameters**

-   `channels (Union[str, List[str]])`: A master list of channel names that defines the mapping from name to index (0-based). This can also be a string key for dynamically importing this master list.
-   `strict_mode (bool, optional)`: If `True`, all channel names present in the input signal's `channels` list *must* exist in the master `channels` list; otherwise, an error will be raised. If `False`, input channel names not found in the master list are silently ignored. Defaults to `False`.
-   `source (str, optional)`: The key in the input dictionary for the signal to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary for the transformed signal. Defaults to `None`.

**Usage Example**

~~~python
# Assume 'master_channel_list' is defined, e.g.:
# master_channel_list = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'Oz']
# Assume 'results' is a dict like:
# results = {
#     'eeg': {
#         'data': np.random.randn(3, 50),   # Data for Cz, Fz, Oz
#         'channels': ['Cz', 'Fz', 'Oz'],   # Channels to be converted
#         'freq': 100.0
#     }
# }
# Initialize ToIndexChannels
t = ToIndexChannels(channels=master_channel_list, strict_mode=False, source='eeg', target='eeg_indexed')
# Apply the transform
processed_results = t(results)
# processed_results['eeg_indexed']['channels'] will be [3, 2, 5]
# (Indices of Cz, Fz, Oz in master_channel_list)
# The 'data' array remains unchanged.
~~~

[`Back to Top`](#tyeedatasettransform)

## Compose

~~~python
class Compose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], source: Optional[str] = None, target: Optional[str] = None):
~~~

A transform that composes a list of other transforms, applying them sequentially.
The `source` and `target` parameters are for the `Compose` instance itself, defining where it reads from and writes to when called on a results dictionary.
Sub-transforms provided in the `transforms` list **must be initialized without `source` or `target` arguments**, as they operate on the data stream passed through the `Compose` pipeline. An attempt to initialize `Compose` with sub-transforms that have `source` or `target` set will raise a `ValueError` (as per the provided implementation).

**Parameters**

-   `transforms (List[BaseTransform])`: A list of transform instances to be applied in sequence. Each transform in this list should be initialized without `source` or `target`.
-   `source (Optional[str])`: The key in the input dictionary from which the `Compose` transform reads its initial data. Defaults to `None`.
-   `target (Optional[str])`: The key in the output dictionary where the `Compose` transform writes the final processed data. If `None`, it defaults to the `source` key. Defaults to `None`.

**Internal `transform` Logic Note:**
The `Compose` class's `transform` method (as per your provided code) iterates through its sub-transforms. For each sub-transform `t`, it effectively executes `result = t(result)` (since `t.source` and `t.target` are `None` due to the `__init__` check, causing the `else` branch of the conditional `current_data_item = t_sub({'data': current_data_item})['data'] if t_sub.source or t_sub.target else t_sub(current_data_item)` to be taken). This relies on the sub-transform's `__call__` method being able to handle the raw data (which, in this context, is typically the signal dictionary) directly.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, PickChannels, CommonAverageRef, Compose) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import PickChannels, CommonAverageRef, Compose

# 1. Prepare an example 'results' dictionary with initial signal data
results = {
    'raw_eeg': {
        'data': np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]), # 4 channels, 3 time points
        'channels': ['CH1', 'CH2', 'CH3', 'CH4'],
        'freq': 100.0
    }
}

# 2. Instantiate sub-transforms (initialized without source/target)
pick_specific_channels = PickChannels(channels=['CH1', 'CH3', 'CH4'])
apply_car_transform = CommonAverageRef(axis=0, mode='mean')

# 3. Instantiate the Compose transform
#    'source' and 'target' are for the Compose instance itself.
pipeline = Compose(
    transforms=[
        pick_specific_channels,
        apply_car_transform
    ],
    source='raw_eeg',
    target='processed_eeg'
)

# 4. Apply the composed transform
processed_results = pipeline(results)

# 5. 'processed_results' will now contain the 'processed_eeg' key with the transformed data.
# Example content of processed_results['processed_eeg']:
# {
#   'data': np.array([[-5., -5., -5.], [ 1.,  1.,  1.], [ 4.,  4.,  4.]]),
#   'channels': ['CH1', 'CH3', 'CH4'],
#   'freq': 100.0
# }
# print(processed_results['processed_eeg'])
~~~

[`Back to Top`](#tyeedatasettransform)

## UniToBiTransform

~~~python
class UniToBiTransform(BaseTransform):
    def __init__(self, target_channels: Union[str, List[str]], source: str = None, target: str = None):
~~~

Converts unipolar signals to bipolar signals. This is achieved by calculating the difference between adjacent channels as specified in the `target_channels` list. Each string in `target_channels` should define a pair of unipolar channels to be converted into a single bipolar channel (e.g., "Fp1-F7"). If any specified unipolar channel is not found in the input signal, a `ValueError` is raised.

**Parameters**

-   `target_channels (Union[str, List[str]])`: A list of strings, where each string specifies a bipolar channel pair in the format "ChannelA-ChannelB". Alternatively, this can be a string key used to dynamically import a list of such channel pair strings (e.g., from `dataset.constants`).
-   `source (str, optional)`: The key in the input dictionary that holds the unipolar signal data (as a signal dictionary with 'data', 'channels', and optionally 'freq'). Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary where the transformed bipolar signal data (as a new signal dictionary) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, UniToBiTransform) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import UniToBiTransform

# 1. Prepare an example 'results' dictionary with initial unipolar signal data
results = {
    'unipolar_eeg': {
        'data': np.array([[1.0, 1.5, 2.0], [0.5, 0.8, 1.2], [2.0, 2.2, 2.5], [1.0, 1.0, 1.0]]),
        'channels': ['Fp1', 'F7', 'Fp2', 'F8'], # Unipolar channels
        'freq': 250.0
    }
}

# 2. Define the target bipolar channel pairs
bipolar_pairs = ["Fp1-F7", "Fp2-F8"]

# 3. Instantiate the UniToBiTransform
#    'source' and 'target' are for the UniToBiTransform instance itself.
transformer = UniToBiTransform(
    target_channels=bipolar_pairs,
    source='unipolar_eeg',
    target='bipolar_eeg'
)

# 4. Apply the transform
processed_results = transformer(results)

# 5. 'processed_results' will now contain the 'bipolar_eeg' key.
#    processed_results['bipolar_eeg']['data'] will contain the differences:
#    Fp1-F7: [1.0-0.5, 1.5-0.8, 2.0-1.2] = [0.5, 0.7, 0.8]
#    Fp2-F8: [2.0-1.0, 2.2-1.0, 2.5-1.0] = [1.0, 1.2, 1.5]
#    processed_results['bipolar_eeg']['channels'] will be ["Fp1-F7", "Fp2-F8"].
# Example content of processed_results['bipolar_eeg']:
# {
#   'data': np.array([[0.5, 0.7, 0.8], [1.0, 1.2, 1.5]]),
#   'channels': ["Fp1-F7", "Fp2-F8"],
#   'freq': 250.0
# }
# print(processed_results['bipolar_eeg'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Crop

~~~python
class Crop(BaseTransform):
    def __init__(self, crop_left=0, crop_right=0, axis=-1, source=None, target=None):
~~~

A general-purpose transform for cropping signals. It can crop a specified number of points from the left (start) and/or right (end) of a signal along a given axis, typically the time axis. This transform is applicable to any signal.

**Parameters**

-   `crop_left (int)`: The number of points to crop from the left (beginning) of the signal. Defaults to `0`.
-   `crop_right (int)`: The number of points to crop from the right (end) of the signal. If `0` or negative, no cropping is performed from the right. Defaults to `0`.
-   `axis (int)`: The axis along which to perform the cropping. This is usually the time axis. Defaults to `-1` (the last axis).
-   `source (str, optional)`: The key in the input dictionary that holds the signal data (as a signal dictionary with 'data', 'channels', etc.) to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary where the transformed (cropped) signal data will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Crop) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Crop

# 1. Prepare an example 'results' dictionary with initial signal data
results = {
    'raw_signal': {
        'data': np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]), # 2 channels, 10 time points
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Instantiate the Crop transform
#    Crop 2 points from the left and 3 points from the right along the last axis (time).
cropper = Crop(crop_left=2, crop_right=3, axis=-1, source='raw_signal', target='cropped_signal')

# 3. Apply the transform
processed_results = cropper(results)

# 4. 'processed_results' will now contain the 'cropped_signal' key.
#    The data will be cropped:
#    Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    After crop_left=2: [3, 4, 5, 6, 7, 8, 9, 10]
#    After crop_right=3 (from original end): [3, 4, 5, 6, 7]
#    So, data for CH1: [3, 4, 5, 6, 7]
#    So, data for CH2: [13, 14, 15, 16, 17]
#    The shape will be (2, 5).
# Example content of processed_results['cropped_signal']:
# {
#   'data': np.array([[ 3,  4,  5,  6,  7],
#                      [13, 14, 15, 16, 17]]),
#   'channels': ['CH1', 'CH2'],
#   'freq': 100.0
# }
# print(processed_results['cropped_signal'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Filter

~~~python
class Filter(BaseTransform):
    def __init__(self, l_freq=None, h_freq=None, filter_length="auto", l_trans_bandwidth="auto", h_trans_bandwidth="auto", method="fir", iir_params=None, phase="zero", fir_window="hamming", fir_design="firwin", pad="reflect_limited", source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a filter to the signal data, leveraging `mne.filter.filter_data`. This can be configured as a band-pass, low-pass, or high-pass filter.

**Parameters**

-   `l_freq (float | None)`: Lower pass-band edge. If `None`, it acts as a low-pass filter.
-   `h_freq (float | None)`: Upper pass-band edge. If `None`, it acts as a high-pass filter.
-   `filter_length (str | int)`: Length of the FIR filter to use (if applicable). Defaults to `"auto"`.
-   `l_trans_bandwidth (str | float)`: Width of the lower transition band. Defaults to `"auto"`.
-   `h_trans_bandwidth (str | float)`: Width of the upper transition band. Defaults to `"auto"`.
-   `method (str)`: Filtering method, either `"fir"` or `"iir"`. Defaults to `"fir"`.
-   `iir_params (dict | None)`: Dictionary of parameters to use for IIR filtering. Defaults to `None`.
-   `phase (str)`: Phase of the filter, either `"zero"` or `"zero-double"`. Defaults to `"zero"`.
-   `fir_window (str)`: The window to use in FIR design. Defaults to `"hamming"`.
-   `fir_design (str)`: Method for FIR design, e.g., `"firwin"`. Defaults to `"firwin"`.
-   `pad (str)`: The type of padding to use. Defaults to `"reflect_limited"`.
-   `source (Optional[str])`: The key for the input signal dictionary. Defaults to `None`.
-   `target (Optional[str])`: The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Filter) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Filter

# 1. Prepare an example 'results' dictionary
results = {
    'raw_signal': {
        'data': np.random.randn(2, 1000), # 2 channels, 1000 time points
        'freq': 250.0, # Sampling frequency
        'channels': ['CH1', 'CH2']
    }
}

# 2. Instantiate a band-pass Filter (e.g., 1 Hz to 40 Hz)
bandpass_filter = Filter(l_freq=1.0, h_freq=40.0, method="fir", source='raw_signal', target='filtered_signal')

# 3. Apply the transform
processed_results = bandpass_filter(results)

# 4. 'processed_results' will now contain 'filtered_signal' with the band-pass filtered data.
#    The 'data' array within processed_results['filtered_signal'] will be modified.
# print(processed_results['filtered_signal']['data'])
~~~

---
[`Back to Top`](#tyeedatasettransform)

## NotchFilter

~~~python
class NotchFilter(BaseTransform):
    def __init__(self, freqs: List[float], filter_length="auto", notch_widths=None, trans_bandwidth=1, method="fir", iir_params=None, mt_bandwidth=None, p_value=0.05, phase="zero", fir_window="hamming", fir_design="firwin", pad="reflect_limited", source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a notch filter to the signal data to remove specific frequency components (e.g., power line noise), using `mne.filter.notch_filter`.

**Parameters**

-   `freqs (List[float])`: Frequencies to notch out.
-   `filter_length (str | int)`: Length of the FIR filter to use. Defaults to `"auto"`.
-   `notch_widths (float | List[float] | None)`: Width of the notch for each frequency. If `None`, it is set to `freqs / 200`.
-   `trans_bandwidth (float)`: Width of the transition band. Defaults to `1`.
-   `method (str)`: Filtering method, e.g., `"fir"`, `"iir"`, or `"spectrum_fit"`. Defaults to `"fir"`.
-   `iir_params (dict | None)`: Parameters for IIR filtering. Defaults to `None`.
-   `mt_bandwidth (float | None)`: Multitaper bandwidth (if `method='spectrum_fit'`).
-   `p_value (float)`: P-value for detecting significant peaks (if `method='spectrum_fit'`).
-   `phase (str)`: Phase of the filter. Defaults to `"zero"`.
-   `fir_window (str)`: Window for FIR design. Defaults to `"hamming"`.
-   `fir_design (str)`: Method for FIR design. Defaults to `"firwin"`.
-   `pad (str)`: Type of padding. Defaults to `"reflect_limited"`.
-   `source (Optional[str])`: The key for the input signal dictionary. Defaults to `None`.
-   `target (Optional[str])`: The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, NotchFilter) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import NotchFilter

# 1. Prepare an example 'results' dictionary
results = {
    'noisy_signal': {
        'data': np.random.randn(2, 1000), # 2 channels, 1000 time points
        'freq': 250.0, # Sampling frequency
        'channels': ['CH1', 'CH2']
    }
}

# 2. Instantiate a NotchFilter (e.g., to remove 50 Hz and 100 Hz noise)
powerline_filter = NotchFilter(freqs=[50.0, 100.0], source='noisy_signal', target='denoised_signal')

# 3. Apply the transform
processed_results = powerline_filter(results)

# 4. 'processed_results' will now contain 'denoised_signal' with the specified frequencies attenuated.
# print(processed_results['denoised_signal']['data'])
~~~

---
[`Back to Top`](#tyeedatasettransform)

## Cheby2Filter

~~~python
class Cheby2Filter(BaseTransform):
    def __init__(self, l_freq, h_freq, order=6, rp=0.1, rs=60, btype='bandpass', source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a Chebyshev type II digital or analog filter to the signal data using `scipy.signal.cheby2` and `scipy.signal.filtfilt` for zero-phase filtering.

**Parameters**

-   `l_freq (float)`: Lower critical frequency.
-   `h_freq (float)`: Upper critical frequency.
-   `order (int)`: The order of the filter. Defaults to `6`.
-   `rp (float)`: For Chebyshev type I filters, the maximum ripple allowed below unity gain in the passband. Not used by Cheby2. Defaults to `0.1`.
-   `rs (float)`: For Chebyshev type II filters, the minimum attenuation required in the stop band. Defaults to `60` (dB).
-   `btype (str)`: The type of filter. Options include `'bandpass'`, `'lowpass'`, `'highpass'`, `'bandstop'`. Defaults to `'bandpass'`.
-   `source (Optional[str])`: The key for the input signal dictionary. Defaults to `None`.
-   `target (Optional[str])`: The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Cheby2Filter) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Cheby2Filter

# 1. Prepare an example 'results' dictionary
results = {
    'raw_data': {
        'data': np.random.randn(3, 1280), # 3 channels, 1280 time points
        'freq': 500.0, # Sampling frequency
        'channels': ['EEG1', 'EEG2', 'EEG3']
    }
}

# 2. Instantiate a Cheby2Filter (e.g., bandpass between 0.5 Hz and 45 Hz)
cheby_filter = Cheby2Filter(l_freq=0.5, h_freq=45.0, order=5, rs=50, source='raw_data', target='cheby_filtered_data')

# 3. Apply the transform
processed_results = cheby_filter(results)

# 4. 'processed_results' will now contain 'cheby_filtered_data' with the filtered signal.
# print(processed_results['cheby_filtered_data']['data'])
~~~

---
[`Back to Top`](#tyeedatasettransform)

## Detrend

~~~python
class Detrend(BaseTransform):
    def __init__(self, axis=-1, source=None, target=None):
~~~

Removes the linear trend from the signal data along a specified axis, using `scipy.signal.detrend`.

**Parameters**

-   `axis (int)`: The axis along which to detrend the data. Defaults to `-1` (the last axis).
-   `source (str, optional)`: The key for the input signal dictionary. Defaults to `None`.
-   `target (str, optional)`: The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Detrend) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Detrend

# 1. Create a signal with a linear trend
sfreq = 100.0
time = np.arange(0, 10, 1/sfreq) # 10 seconds of data
trend = 0.5 * time # Linear trend
signal_component = np.sin(2 * np.pi * 5 * time) # 5 Hz sine wave
original_data_ch1 = signal_component + trend
original_data_ch2 = np.cos(2 * np.pi * 10 * time) + 0.3 * time # Another channel with a trend

results = {
    'trended_signal': {
        'data': np.array([original_data_ch1, original_data_ch2]),
        'freq': sfreq,
        'channels': ['CH1_trend', 'CH2_trend']
    }
}

# 2. Instantiate the Detrend transform
detrender = Detrend(axis=-1, source='trended_signal', target='detrended_signal')

# 3. Apply the transform
processed_results = detrender(results)

# 4. 'processed_results' will now contain 'detrended_signal' where the linear trend has been removed.
# print(processed_results['detrended_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ForEach

~~~python
class ForEach(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], source=None, target=None):
~~~

Applies a sequence of transforms to each segment along the first dimension of the input data. This is suitable for 2D or 3D data (e.g., shape=(N, ...), where N is the number of segments, epochs, or channels). The sub-transforms provided in the `transforms` list must be initialized without `source` or `target` arguments, as they operate on individual segments passed internally.

**Parameters**

-   `transforms (List[BaseTransform])`: A list of transform instances to be applied sequentially to each segment. Each transform in this list should be initialized without `source` or `target`.
-   `source (str, optional)`: The key in the input dictionary that holds the signal data (as a signal dictionary with 'data', 'channels', etc.) to be processed. The 'data' under this key is expected to be at least 2D. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary where the processed signal data (with each segment transformed) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Crop, Detrend, ForEach) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Crop, Detrend, ForEach

# 1. Prepare an example 'results' dictionary with 3D data (e.g., epochs x channels x time)
results = {
    'epoched_signal': {
        'data': np.random.randn(2, 3, 100), # 2 epochs, 3 channels, 100 time points per epoch
        'channels': ['CH1', 'CH2', 'CH3'], # Channels are consistent across epochs
        'freq': 100.0
    }
}

# 2. Instantiate sub-transforms to be applied to each epoch.
#    These are initialized without source/target.
crop_each_epoch = Crop(crop_left=10, crop_right=10, axis=-1) # Crop time axis of each epoch
detrend_each_epoch = Detrend(axis=-1) # Detrend time axis of each epoch

# 3. Instantiate the ForEach transform.
#    It will iterate over the first dimension of 'epoched_signal']['data'] (the epochs).
process_each_epoch = ForEach(
    transforms=[
        crop_each_epoch,
        detrend_each_epoch
    ],
    source='epoched_signal',
    target='processed_epochs'
)

# 4. Apply the ForEach transform
processed_results = process_each_epoch(results)

# 5. 'processed_results' will now contain 'processed_epochs'.
#    The 'data' in 'processed_epochs' will have shape (2, 3, 80) because:
#    - ForEach iterates 2 times (once for each epoch).
#    - Each epoch (3x100) is first cropped to (3x80) by crop_each_epoch.
#    - Then, each cropped epoch (3x80) is detrended by detrend_each_epoch.
#    - The resulting (3x80) segments are stacked back, resulting in (2, 3, 80).
#    The 'channels' and 'freq' would typically be preserved from the input signal dictionary.
#
# Example content of processed_results['processed_epochs']:
# {
#   'data': np.ndarray of shape (2, 3, 80), # Data for 2 epochs, 3 channels, 80 time points
#   'channels': ['CH1', 'CH2', 'CH3'],
#   'freq': 100.0
# }
# print(processed_results['processed_epochs']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## Lambda

~~~python
class Lambda(BaseTransform):
    def __init__(self, lambd: Callable, source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a user-defined lambda function or any callable to the 'data' field within the input signal dictionary. This allows for flexible, custom operations on the signal data.

**Parameters**

-   `lambd (Callable)`: A callable (e.g., a lambda function or a regular function) that takes the signal data (e.g., a NumPy array) as input and returns the transformed data.
-   `source (str, optional)`: The key in the input dictionary that holds the signal dictionary (containing 'data', 'channels', etc.) to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary where the transformed signal dictionary will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Lambda) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from typing import Callable, Optional # For Callable, Optional type hints
# from tyee.dataset.transform import Lambda

# 1. Prepare an example 'results' dictionary
results = {
    'raw_signal': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 2 channels, 3 time points
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Define a lambda function for a custom operation (e.g., multiply data by 10)
multiply_by_ten = lambda x: x * 10

# 3. Instantiate the Lambda transform
custom_transform = Lambda(
    lambd=multiply_by_ten,
    source='raw_signal',
    target='transformed_signal'
)

# 4. Apply the transform
processed_results = custom_transform(results)

# 5. 'processed_results' will now contain 'transformed_signal'.
#    The 'data' in 'transformed_signal' will be the original data multiplied by 10.
# Example content of processed_results['transformed_signal']:
# {
#   'data': np.array([[10, 20, 30], [40, 50, 60]]),
#   'channels': ['CH1', 'CH2'],
#   'freq': 100.0
# }
# print(processed_results['transformed_signal'])

# Example with a different lambda, e.g., select first channel
select_first_channel_data = lambda x: x[0:1, :] # Keep it as a 2D array
channel_selector = Lambda(lambd=select_first_channel_data, source='raw_signal', target='first_channel_data')
selected_channel_results = channel_selector(results)
# selected_channel_results['first_channel_data']['data'] would be np.array([[1, 2, 3]])
# Note: This lambda only changes 'data'. The 'channels' list in the signal dictionary
# would still be ['CH1', 'CH2'] unless the lambda or another transform updates it.
# print(selected_channel_results['first_channel_data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Mapping

~~~python
class Mapping(BaseTransform):
    def __init__(self, mapping: dict, source: str = None, target: str = None):
~~~

Applies a dictionary mapping to the values in the 'data' field of the input signal dictionary. If the 'data' is a scalar, it's mapped directly. If it's a NumPy array or other iterable, each element is mapped individually, preserving the original shape for NumPy arrays.

**Parameters**

-   `mapping (dict)`: A dictionary where keys are the original values and values are the target values they should be mapped to.
-   `source (str, optional)`: The key in the input dictionary that holds the signal dictionary (containing 'data', etc.) to be transformed. Defaults to `None`.
-   `target (str, optional)`: The key in the output dictionary where the transformed signal dictionary (with mapped 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Mapping) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Mapping

# 1. Prepare an example 'results' dictionary
results_scalar = {
    'label_data': {
        'data': 1, # Scalar data
        'channels': ['EventLabel'], # Conceptual
    }
}
results_array = {
    'categorical_data': {
        'data': np.array([[0, 1, 2], [2, 0, 1]]), # Array data
        'channels': ['FeatureSet1', 'FeatureSet2'], # Conceptual
    }
}

# 2. Define a mapping dictionary
category_mapping = {
    0: 100, # Map 0 to 100
    1: 200, # Map 1 to 200
    2: 300  # Map 2 to 300
}

# 3. Instantiate the Mapping transform for scalar data
map_transform_scalar = Mapping(
    mapping=category_mapping,
    source='label_data',
    target='mapped_label'
)

# 4. Instantiate the Mapping transform for array data
map_transform_array = Mapping(
    mapping=category_mapping,
    source='categorical_data',
    target='mapped_array'
)

# 5. Apply the transform to scalar data
processed_results_scalar = map_transform_scalar(results_scalar)
# processed_results_scalar['mapped_label']['data'] will be 200 (since original data was 1)

# 6. Apply the transform to array data
processed_results_array = map_transform_array(results_array)
# processed_results_array['mapped_array']['data'] will be:
# np.array([[100, 200, 300], [300, 100, 200]])

# print("Mapped scalar data:", processed_results_scalar['mapped_label']['data'])
# print("Mapped array data:\n", processed_results_array['mapped_array']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Concat

~~~python
class Concat(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
~~~

Concatenates the 'data' arrays from a list of input signal dictionaries along a specified axis. If the `source` argument is a list of keys, the transform expects to receive a corresponding list of signal dictionaries. It attempts to merge 'channels' lists and retain the 'freq' from the first signal if available. If `source` is a single string, it implies the input signal dictionary itself contains data that might be suitable for some form of concatenation (though the provided `transform` code primarily handles the list-of-sources case for concatenation).

**Parameters**

-   `axis (int)`: The axis along which the 'data' arrays will be concatenated. Defaults to `0`.
-   `source (Optional[Union[str, List[str]]])`: A list of string keys identifying the input signal dictionaries in the main results dictionary, or a single string key. If a list, the corresponding signal dictionaries are processed. Defaults to `None`.
-   `target (Optional[str])`: The key in the output dictionary where the new signal dictionary (containing the concatenated 'data', merged 'channels', and 'freq') will be stored. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Concat) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Concat

# 1. Prepare an example 'results' dictionary with multiple signal sources
results = {
    'signal_A': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # Shape (2, 3)
        'channels': ['A_CH1', 'A_CH2'],
        'freq': 100.0
    },
    'signal_B': {
        'data': np.array([[7, 8, 9]]), # Shape (1, 3)
        'channels': ['B_CH1'],
        'freq': 100.0 # Assuming same frequency
    },
    'signal_C': {
        'data': np.array([[10, 11, 12], [13, 14, 15]]), # Shape (2,3)
        'channels': 'C_CH_Group', # Example of non-list channels
        'freq': 100.0
    }
}

# 2. Instantiate the Concat transform to concatenate along axis 0 (channels)
#    The 'source' is a list of keys.
concatenator = Concat(
    axis=0,
    source=['signal_A', 'signal_B', 'signal_C'],
    target='concatenated_signal'
)

# 3. Apply the transform
#    The __call__ method of BaseTransform should handle collecting signals from source keys.
processed_results = concatenator(results)

# 4. 'processed_results' will now contain 'concatenated_signal'.
#    'data' will be the concatenation of signal_A['data'] and signal_B['data'] along axis 0.
#    Shape: (2+1+2, 3) = (5, 3)
#    'channels' will be ['A_CH1', 'A_CH2', 'B_CH1', 'C_CH_Group']
#    'freq' will be 100.0
#
# Example content of processed_results['concatenated_signal']:
# {
#   'data': np.array([[ 1,  2,  3],
#                      [ 4,  5,  6],
#                      [ 7,  8,  9],
#                      [10, 11, 12],
#                      [13, 14, 15]]),
#   'channels': ['A_CH1', 'A_CH2', 'B_CH1', 'C_CH_Group'],
#   'freq': 100.0
# }
# print(processed_results['concatenated_signal'])
~~~

---
[`Back to Top`](#tyeedatasettransform)

## Stack

~~~python
class Stack(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
~~~

Stacks the 'data' arrays from a list of input signal dictionaries along a new axis. If the `source` argument is a list of keys, the transform expects to receive a corresponding list of signal dictionaries. The 'data' arrays from these dictionaries must have compatible shapes for stacking. If `source` is a single string, it implies the input signal dictionary itself might be processed (though the provided `transform` code primarily handles the list-of-sources case for stacking).

**Parameters**

-   `axis (int)`: The axis along which the 'data' arrays will be stacked. This will be a new axis in the resulting 'data' array. Defaults to `0`.
-   `source (Optional[Union[str, List[str]]])`: A list of string keys identifying the input signal dictionaries in the main results dictionary, or a single string key. If a list, the corresponding signal dictionaries are processed. Defaults to `None`.
-   `target (Optional[str])`: The key in the output dictionary where the new signal dictionary (containing the stacked 'data') will be stored. Metadata like 'channels' and 'freq' are not explicitly merged or retained by this transform's current implementation, only 'data' is returned in the new dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume necessary classes (BaseTransform, Stack) and numpy (as np) are imported.
# e.g.:
# import numpy as np
# from tyee.dataset.transform import Stack

# 1. Prepare an example 'results' dictionary with multiple signal sources
#    Data arrays should have the same shape for stacking.
results = {
    'epoch_1': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # Shape (2 channels, 3 time points)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    },
    'epoch_2': {
        'data': np.array([[7, 8, 9], [10, 11, 12]]), # Shape (2 channels, 3 time points)
        'channels': ['CH1', 'CH2'], # Assuming same channels for simplicity
        'freq': 100.0
    },
    'epoch_3': {
        'data': np.array([[13,14,15], [16,17,18]]), # Shape (2 channels, 3 time points)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Instantiate the Stack transform to stack along a new first axis (axis=0)
#    This will create a new dimension for epochs.
stacker = Stack(
    axis=0,
    source=['epoch_1', 'epoch_2', 'epoch_3'],
    target='stacked_epochs'
)

# 3. Apply the transform
processed_results = stacker(results)

# 4. 'processed_results' will now contain 'stacked_epochs'.
#    'data' will be the stacking of epoch_1['data'], epoch_2['data'], and epoch_3['data'].
#    Original shape of each data: (2, 3)
#    New shape after stacking along axis=0: (3 epochs, 2 channels, 3 time points)
#
# Example content of processed_results['stacked_epochs']:
# {
#   'data': np.array([[[ 1,  2,  3], [ 4,  5,  6]],
#                      [[ 7,  8,  9], [10, 11, 12]],
#                      [[13, 14, 15], [16, 17, 18]]])
# }
# Note: The current Stack transform implementation only returns {'data': stacked_data}.
# It does not propagate 'channels' or 'freq'.
# print(processed_results['stacked_epochs']['data'].shape)
# print(processed_results['stacked_epochs'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ZScoreNormalize

~~~python
class ZScoreNormalize(BaseTransform):
    def __init__(
        self, 
        mean: Optional[np.ndarray] = None, 
        std: Optional[np.ndarray] = None, 
        axis: Optional[int] = None, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

Performs Z-score normalization on the 'data' field of a signal dictionary. This can be done along a specified axis, and optionally with a user-provided mean and standard deviation. If a custom mean/std is provided as a 1D array and an axis is specified, it will be reshaped to allow for broadcasting. If the standard deviation is zero, it's treated as one to avoid division by zero.

**Parameters**

- **mean** (`Optional[np.ndarray]`): A pre-calculated mean to use for normalization. If `None`, the mean is calculated from the data. Defaults to `None`.
- **std** (`Optional[np.ndarray]`): A pre-calculated standard deviation. If `None`, the standard deviation is calculated from the data. Defaults to `None`.
- **axis** (`Optional[int]`): The axis along which to compute the mean and standard deviation if they are not provided. If `None`, normalization is performed over the entire data. Defaults to `None`.
- **epsilon** (`float`): A small value added to the standard deviation to prevent division by zero. Defaults to `1e-8`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and ZScoreNormalize are imported, and numpy as np
# from tyee.dataset.transform import ZScoreNormalize
# import numpy as np

# 1. Prepare example data
results = {
    'raw_signal': {
        'data': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Instantiate ZScoreNormalize to normalize along rows (axis=1)
normalizer_axis1 = ZScoreNormalize(axis=1, source='raw_signal', target='normalized_axis1')
processed_results_axis1 = normalizer_axis1(results)

# 3. Instantiate ZScoreNormalize with overall normalization (axis=None)
normalizer_overall = ZScoreNormalize(axis=None, source='raw_signal', target='normalized_overall')
processed_results_overall = normalizer_overall(results.copy()) # Use a copy for separate processing

# print("Normalized along axis 1:\n", processed_results_axis1['normalized_axis1']['data'])
# print("Normalized overall:\n", processed_results_overall['normalized_overall']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## MinMaxNormalize

~~~python
class MinMaxNormalize(BaseTransform):
    def __init__(
        self, 
        min: Optional[Union[np.ndarray, float]] = None, 
        max: Optional[Union[np.ndarray, float]] = None, 
        axis: Optional[int] = None, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

Performs min-max normalization on the 'data' field of a signal dictionary, scaling the data to a range (typically 0 to 1). It supports normalization along a specified axis and can use custom minimum and maximum values. If custom min/max are provided as 1D arrays and an axis is specified, they will be expanded to match the data's dimensions for broadcasting.

**Parameters**

- **min** (`Optional[Union[np.ndarray, float]]`): A pre-calculated minimum value or array. If `None`, the minimum is calculated from the data. Defaults to `None`.
- **max** (`Optional[Union[np.ndarray, float]]`): A pre-calculated maximum value or array. If `None`, the maximum is calculated from the data. Defaults to `None`.
- **axis** (`Optional[int]`): The axis along which to compute the min and max if they are not provided. If `None`, normalization is performed over the entire data. Defaults to `None`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and MinMaxNormalize are imported, and numpy as np
# from tyee.dataset.transform import MinMaxNormalize
# import numpy as np

# 1. Prepare example data
results = {
    'raw_data': {
        'data': np.array([[10., 20., 30.], [0., 50., 100.]]),
        'channels': ['A', 'B'],
        'freq': 100.0
    }
}

# 2. Instantiate MinMaxNormalize to normalize along columns (axis=0)
minmax_axis0 = MinMaxNormalize(axis=0, source='raw_data', target='minmax_axis0')
processed_axis0 = minmax_axis0(results)

# 3. Instantiate MinMaxNormalize for overall normalization
minmax_overall = MinMaxNormalize(source='raw_data', target='minmax_overall')
processed_overall = minmax_overall(results.copy())

# print("Normalized along axis 0:\n", processed_axis0['minmax_axis0']['data'])
# print("Normalized overall:\n", processed_overall['minmax_overall']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## QuantileNormalize

~~~python
class QuantileNormalize(BaseTransform):
    def __init__(
        self, 
        q: float = 0.95, 
        axis: Optional[int] = -1, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

Performs quantile normalization on the 'data' field. This method normalizes the data by dividing it by a specified quantile of its absolute values. It can be applied along a specific axis.

**Parameters**

- **q** (`float`): The quantile to compute (between 0 and 1). Defaults to `0.95`.
- **axis** (`Optional[int]`): The axis along which to compute the quantile. Defaults to `-1` (last axis).
- **epsilon** (`float`): A small value added to the quantile value to prevent division by zero. Defaults to `1e-8`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and QuantileNormalize are imported, and numpy as np
# from tyee.dataset.transform import QuantileNormalize
# import numpy as np

# 1. Prepare example data
results = {
    'signal': {
        'data': np.array([[-10., 0., 10., 20., 100.], [1., 2., 3., 4., 5.]]),
        'channels': ['S1', 'S2'],
        'freq': 100.0
    }
}

# 2. Instantiate QuantileNormalize using the 90th percentile along rows (axis=1)
quantile_norm = QuantileNormalize(q=0.9, axis=1, source='signal', target='q_normalized_signal')
processed_results = quantile_norm(results)

# print("Quantile normalized signal (q=0.9, axis=1):\n", processed_results['q_normalized_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## RobustNormalize

~~~python
class RobustNormalize(BaseTransform):
    def __init__(
        self,
        median: Optional[np.ndarray] = None,
        iqr: Optional[np.ndarray] = None,
        quantile_range: tuple = (25.0, 75.0),
        axis: Optional[int] = None,
        epsilon: float = 1e-8,
        unit_variance: bool = False,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

Performs robust normalization using the formula `(x - median) / IQR`, where IQR (Interquartile Range) is `q_max - q_min`. This method is resistant to outliers. It supports normalization along a specified axis and can optionally scale the result to unit variance. Custom median and IQR values can also be provided.

**Parameters**

- **median** (`Optional[np.ndarray]`): A pre-calculated median. If `None`, it's computed from the data. Defaults to `None`.
- **iqr** (`Optional[np.ndarray]`): A pre-calculated Interquartile Range. If `None`, it's computed from the data using `quantile_range`. Defaults to `None`.
- **quantile_range** (`tuple`): Tuple of two floats `(q_min, q_max)` representing the lower and upper percentile for IQR calculation (0-100). Defaults to `(25.0, 75.0)`.
- **axis** (`Optional[int]`): The axis along which to compute median and IQR. If `None`, computations are done over the entire data. Defaults to `None`.
- **epsilon** (`float`): A small value added to the IQR to prevent division by zero. Defaults to `1e-8`.
- **unit_variance** (`bool`): If `True`, scales the output so that the IQR corresponds to the range between `norm.ppf(q_min/100.0)` and `norm.ppf(q_max/100.0)` of a standard normal distribution, effectively making the variance of the central part of the data closer to 1. Defaults to `False`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and RobustNormalize are imported, and numpy as np
# from tyee.dataset.transform import RobustNormalize
# import numpy as np

# 1. Prepare example data, including an outlier
results = {
    'measurements': {
        'data': np.array([[1., 2., 3., 4., 5., 100.], [6., 7., 8., 9., 10., -50.]]),
        'channels': ['Sensor1', 'Sensor2']
    }
}

# 2. Instantiate RobustNormalize to normalize along rows (axis=1)
robust_scaler_axis1 = RobustNormalize(axis=1, source='measurements', target='robust_scaled_axis1')
processed_axis1 = robust_scaler_axis1(results)

# 3. Instantiate RobustNormalize for overall normalization with unit_variance
robust_scaler_overall_uv = RobustNormalize(unit_variance=True, source='measurements', target='robust_scaled_overall_uv')
processed_overall_uv = robust_scaler_overall_uv(results.copy())

# print("Robust scaled (axis=1):\n", processed_axis1['robust_scaled_axis1']['data'])
# print("Robust scaled (overall, unit variance):\n", processed_overall_uv['robust_scaled_overall_uv']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Baseline

~~~python
class Baseline(BaseTransform):
    def __init__(
        self,
        baseline_start: Optional[int] = None,
        baseline_end: Optional[int] = None,
        axis: Optional[int] = -1,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

Performs baseline correction on the signal data by subtracting the mean of a specified interval. It supports custom baseline intervals (e.g., the first N samples or any arbitrary range). The `axis` parameter controls how the baseline mean is computed: `-1` (default) for independent baseline per channel, or `None` for a single baseline value across all channels and samples in the baseline period.

**Parameters**

- **baseline_start** (`Optional[int]`): The starting sample index for the baseline period. If `None`, it starts from the beginning (index 0). Defaults to `None`.
- **baseline_end** (`Optional[int]`): The ending sample index (exclusive) for the baseline period. If `None`, it extends to the end of the signal. Defaults to `None`.
- **axis** (`Optional[int]`): The axis along which the mean is calculated for the baseline. If `-1`, the mean is computed for each channel independently over the time samples in the baseline period. If `None`, a single mean is computed over all channels and all time samples in the baseline period. Defaults to `-1`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Baseline are imported, and numpy as np
# from tyee.dataset.transform import Baseline
# import numpy as np

# 1. Prepare example data
data_array = np.array([[1., 2., 10., 11., 12.], [20., 21., 30., 31., 32.]]) # 2 channels, 5 time points
results = {
    'trial_data': {
        'data': data_array,
        'channels': ['ChA', 'ChB'],
        'freq': 100.0
    }
}

# 2. Instantiate Baseline to use the first 2 samples for baseline correction per channel
baseline_corrector = Baseline(baseline_start=0, baseline_end=2, axis=-1, source='trial_data', target='baselined_data')
processed_results = baseline_corrector(results)

# Baseline for ChA: mean([1,2]) = 1.5. Corrected ChA: [-0.5, 0.5, 8.5, 9.5, 10.5]
# Baseline for ChB: mean([20,21]) = 20.5. Corrected ChB: [-0.5, 0.5, 9.5, 10.5, 11.5]
# print("Baselined data (per channel):\n", processed_results['baselined_data']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Mean

~~~python
class Mean(BaseTransform):
    def __init__(self, axis: Optional[Union[int, tuple]] = None, source: Optional[str] = None, target: Optional[str] = None, keepdims: bool = False):
~~~

Calculates the mean of the 'data' field in a signal dictionary along a specified axis or axes. The original signal dictionary structure is preserved, but its 'data' field is replaced by the computed mean.

**Parameters**

- **axis** (`Optional[Union[int, tuple]]`): Axis or axes along which the means are computed. The default is to compute the mean of the flattened array. Defaults to `None`.
- **keepdims** (`bool`): If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. Defaults to `False`.
- 

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Mean are imported, and numpy as np
# from tyee.dataset.transform import Mean
# import numpy as np

# 1. Prepare example data
results = {
    'time_series_data': {
        'data': np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]), # Shape (2, 2, 3)
        'channels': ['Epoch1_CH1', 'Epoch1_CH2', 'Epoch2_CH1', 'Epoch2_CH2'], # Conceptual
        'freq': 100.0
    }
}

# 2. Instantiate Mean to compute the mean along the last axis (axis=-1) with keepdims=True
mean_transform = Mean(axis=-1, keepdims=True, source='time_series_data', target='mean_over_time')
processed_results = mean_transform(results)

# Original data shape (2,2,3)
# Mean along axis=-1 with keepdims=True results in shape (2,2,1)
# Data for first epoch, first channel: mean([1,2,3]) = 2.0
# Data for first epoch, second channel: mean([4,5,6]) = 5.0
# print("Mean over time (keepdims=True):\n", processed_results['mean_over_time']['data'])
# print("Shape after mean transform:", processed_results['mean_over_time']['data'].shape)

# Example of overall mean
overall_mean_transform = Mean(axis=None, source='time_series_data', target='overall_mean')
overall_mean_results = overall_mean_transform(results.copy())
# print("Overall mean:", overall_mean_results['overall_mean']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## OneHotEncode

~~~python
class OneHotEncode(BaseTransform):
    def __init__(self, num: int, source=None, target=None):
~~~

Converts the input data in the 'data' field of a signal dictionary to a one-hot encoded format. The input data can be a scalar integer or an array-like object (list, NumPy array) of integers representing categorical labels. The `num` parameter specifies the total number of classes, determining the length of the one-hot vectors.

**Parameters**

- **num** (`int`): The total number of distinct classes. This will be the dimension of the one-hot encoded vector (e.g., if `num=5`, a label of `2` becomes `[0,0,1,0,0]`).
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing the 'data' to be one-hot encoded). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with one-hot encoded 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and OneHotEncode are imported, and numpy as np
# from tyee.dataset.transform import OneHotEncode
# import numpy as np

# 1. Prepare example data (scalar and array)
results_scalar = {
    'label_scalar': {
        'data': 2, # Scalar label
        'info': 'Sample Label A'
    }
}
results_array = {
    'label_array': {
        'data': np.array([0, 1, 3, 1]), # Array of labels
        'info': 'Sample Labels B'
    }
}

# 2. Define the total number of classes for one-hot encoding
num_classes = 4

# 3. Instantiate OneHotEncode for scalar data
one_hot_encoder_scalar = OneHotEncode(num=num_classes, source='label_scalar', target='encoded_label_scalar')

# 4. Instantiate OneHotEncode for array data
one_hot_encoder_array = OneHotEncode(num=num_classes, source='label_array', target='encoded_label_array')

# 5. Apply the transform to scalar data
processed_scalar = one_hot_encoder_scalar(results_scalar)
# processed_scalar['encoded_label_scalar']['data'] will be np.array([0., 0., 1., 0.])

# 6. Apply the transform to array data
processed_array = one_hot_encoder_array(results_array)
# processed_array['encoded_label_array']['data'] will be:
# np.array([[1., 0., 0., 0.],
#           [0., 1., 0., 0.],
#           [0., 0., 0., 1.],
#           [0., 1., 0., 0.]])

# print("One-hot encoded scalar:\n", processed_scalar['encoded_label_scalar']['data'])
# print("One-hot encoded array:\n", processed_array['encoded_label_array']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Pad

~~~python
class Pad(BaseTransform):
    def __init__(
        self,
        pad_len: int,
        axis: int = 0,
        side: str = 'post',
        mode: str = 'constant',
        constant_values: float = 0,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

Pads the 'data' array within a signal dictionary along a specified axis. This transform allows for padding before, after, or on both sides of the selected axis, using various padding modes supported by `numpy.pad`.

**Parameters**

- **pad_len** (`int`): The number of elements to pad. If `side` is 'both', this number of elements will be added to both sides of the axis.
- **axis** (`int`): The axis along which the padding should be applied. Defaults to `0`.
- **side** (`str`): Specifies whether to pad 'pre' (before), 'post' (after), or on 'both' sides of the axis. Must be one of 'pre', 'post', or 'both'. Defaults to `'post'`.
- **mode** (`str`): The padding mode to use, as defined by `numpy.pad` (e.g., 'constant', 'reflect', 'edge'). Defaults to `'constant'`.
- **constant_values** (`float`): The value to use for padding when `mode` is 'constant'. Defaults to `0`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be padded. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with padded 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Pad are imported, and numpy as np
# from tyee.dataset.transform import Pad
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'short_signal': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # Shape (2, 3)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Instantiate the Pad transform to add 2 zeros after the data along axis 1 (time)
padder_post = Pad(
    pad_len=2,
    axis=1,
    side='post',
    mode='constant',
    constant_values=0,
    source='short_signal',
    target='padded_signal_post'
)
processed_post = padder_post(results)
# processed_post['padded_signal_post']['data'] will be:
# np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])

# 3. Instantiate Pad to add 1 element (e.g., edge value) before data along axis 0 (channels)
padder_pre_edge = Pad(
    pad_len=1,
    axis=0,
    side='pre',
    mode='edge', # Edge padding uses the edge values
    source='short_signal',
    target='padded_signal_pre_edge'
)
processed_pre_edge = padder_pre_edge(results.copy()) # Use a copy for separate processing
# processed_pre_edge['padded_signal_pre_edge']['data'] will be:
# np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]) if data was [[1,2,3],[4,5,6]]

# print("Padded post (axis=1):\n", processed_post['padded_signal_post']['data'])
# print("Padded pre with edge (axis=0):\n", processed_pre_edge['padded_signal_pre_edge']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Resample

~~~python
class Resample(BaseTransform):
    def __init__(self,
                 desired_freq: Optional[int] = None,
                 axis: int = -1,
                 window: str = "auto",
                 n_jobs: Optional[int] = None, # Corrected from n_jobs: Optional[int] = None, in provided code
                 pad: str = "auto",
                 npad: str = 'auto', # This was npad in your code, MNE uses n_pad_type or similar, but I'll stick to your variable name
                 method: str = "fft", # MNE uses 'fft' or 'polyphase' for its resample function
                 verbose: Optional[bool] = None,
                 source: Optional[str] = None,
                 target: Optional[str] = None):
~~~

Resamples the 'data' in a signal dictionary to a new `desired_freq` using `mne.filter.resample`. This transform is suitable for 2D signals (e.g., channels x time). If the current frequency is already the desired frequency, or if `desired_freq` is `None`, no resampling is performed.

**Parameters**

- **desired_freq** (`Optional[int]`): The target sampling frequency. If `None` or equal to the current frequency, the signal is returned unchanged.
- **axis** (`int`): The axis along which to resample the data (typically the time axis). Defaults to `-1`.
- **window** (`str`): The window to use in resampling. See `mne.filter.resample` for options. Defaults to `"auto"`.
- **n_jobs** (`Optional[int]`): The number of jobs to run in parallel for resampling. Defaults to `None` (usually meaning 1 job). `mne.filter.resample` uses `n_jobs=1` as default if not specified as `None` for parallel backend.
- **pad** (`str`): The type of padding to use. See `mne.filter.resample`. Defaults to `"auto"`.
- **npad** (`str`): Amount of padding to apply. (Note: `mne.filter.resample` typically uses `npad='auto'` or an integer. The parameter name in MNE might be slightly different, e.g., `n_pad_type` or related to padding length). Defaults to `'auto'`.
- **method** (`str`): The resampling method. Can be `'fft'` (for FFT-based resampling) or `'polyphase'` (for polyphase filtering, often faster for downsampling). `mne.filter.resample` uses this. Defaults to `"fft"`.
- **verbose** (`Optional[bool]`): Controls the verbosity of the MNE resample function. Defaults to `None`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, Resample are imported, and numpy as np.
# from tyee.dataset.transform import Resample
# import numpy as np

# 1. Prepare example data
results = {
    'raw_signal': {
        'data': np.random.randn(5, 1000), # 5 channels, 1000 time points
        'freq': 200.0, # Original sampling frequency
        'channels': [f'CH{i+1}' for i in range(5)]
    }
}

# 2. Instantiate Resample to change frequency to 100 Hz
resampler = Resample(desired_freq=100, source='raw_signal', target='resampled_signal')

# 3. Apply the transform
processed_results = resampler(results)

# 4. 'processed_results' will contain 'resampled_signal'
#    The 'data' will be resampled, and 'freq' will be updated to 100.0.
#    The new number of time points will be approximately 1000 * (100/200) = 500.
# print(f"Original Freq: {results['raw_signal']['freq']}, New Freq: {processed_results['resampled_signal']['freq']}")
# print(f"Original Shape: {results['raw_signal']['data'].shape}, New Shape: {processed_results['resampled_signal']['data'].shape}")
~~~

[`Back to Top`](#tyeedatasettransform)

## Downsample

~~~python
class Downsample(BaseTransform):
    def __init__(self, desired_freq: int, axis: int = -1, source=None, target=None):
~~~

Downsamples the signal 'data' from its current frequency (`cur_freq`) to a `desired_freq` by selecting every Nth sample, where N is `cur_freq // desired_freq`. The 'freq' field in the signal dictionary is updated accordingly. If the current frequency is already the desired frequency or if `cur_freq` is not available, the signal is returned unchanged.

**Parameters**

- **desired_freq** (`int`): The target sampling frequency after downsampling. Must be a divisor of the current frequency for simple decimation.
- **axis** (`int`): The axis along which to downsample (typically the time axis). Defaults to `-1`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, Downsample are imported, and numpy as np.
# from tyee.dataset.transform import Downsample
# import numpy as np

# 1. Prepare example data
results = {
    'high_freq_signal': {
        'data': np.random.randn(3, 1200), # 3 channels, 1200 time points
        'freq': 300.0, # Original sampling frequency
        'channels': ['X', 'Y', 'Z']
    }
}

# 2. Instantiate Downsample to reduce frequency to 100 Hz
downsampler = Downsample(desired_freq=100, source='high_freq_signal', target='downsampled_signal')

# 3. Apply the transform
processed_results = downsampler(results)

# 4. 'processed_results' will contain 'downsampled_signal'
#    The 'data' will be downsampled by taking every 3rd sample (300/100=3).
#    'freq' will be updated to 100.0.
#    New number of time points: 1200 / 3 = 400.
# print(f"Original Freq: {results['high_freq_signal']['freq']}, New Freq: {processed_results['downsampled_signal']['freq']}")
# print(f"Original Shape: {results['high_freq_signal']['data'].shape}, New Shape: {processed_results['downsampled_signal']['data'].shape}")
~~~

[`Back to Top`](#tyeedatasettransform)

## Interpolate

~~~python
class Interpolate(BaseTransform):
    def __init__(self, desired_freq: int, axis: int = -1, kind: str = 'linear', source=None, target=None):
~~~

Upsamples the signal 'data' from its current frequency (`cur_freq`) to a `desired_freq` using interpolation via `scipy.interpolate.interp1d`. The 'freq' field in the signal dictionary is updated. If the current frequency is already the desired frequency or `cur_freq` is not available, the signal is returned unchanged. The current implementation pads the data by appending the last sample before interpolation, which might be a specific strategy for handling boundaries.

**Parameters**

- **desired_freq** (`int`): The target sampling frequency after interpolation.
- **axis** (`int`): The axis along which to interpolate (typically the time axis). Defaults to `-1`.
- **kind** (`str`): Specifies the kind of interpolation as a string (e.g., ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’). Defaults to `'linear'`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, Interpolate are imported, and numpy as np, scipy.interpolate.interp1d
# from tyee.dataset.transform import Interpolate
# import numpy as np
# from scipy.interpolate import interp1d # Needed for the class to run

# 1. Prepare example data
results = {
    'low_freq_signal': {
        'data': np.array([[1., 2., 3.], [10., 11., 12.]]), # 2 channels, 3 time points
        'freq': 50.0, # Original sampling frequency
        'channels': ['SensorA', 'SensorB']
    }
}

# 2. Instantiate Interpolate to increase frequency to 100 Hz using linear interpolation
interpolator = Interpolate(desired_freq=100, kind='linear', source='low_freq_signal', target='interpolated_signal')

# 3. Apply the transform
processed_results = interpolator(results)

# 4. 'processed_results' will contain 'interpolated_signal'
#    The 'data' will be upsampled by a factor of 2 (100/50=2).
#    'freq' will be updated to 100.0.
#    The number of time points will be approximately doubled (minus one due to specific indexing in example).
#    E.g., for [1,2,3] -> ratio=2, old_indices=[0,2,4], new_indices=[0,1,2,3]
#    f(0)=1, f(1)=1.5, f(2)=2, f(3)=2.5
# print(f"Original Freq: {results['low_freq_signal']['freq']}, New Freq: {processed_results['interpolated_signal']['freq']}")
# print(f"Original Shape: {results['low_freq_signal']['data'].shape}, New Shape: {processed_results['interpolated_signal']['data'].shape}")
# print("Interpolated Data:\n", processed_results['interpolated_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Reshape

~~~python
class Reshape(BaseTransform):
    def __init__(self, shape: Tuple[int, ...], source: Optional[str] = None, target: Optional[str] = None):
~~~

A general-purpose transform that reshapes the 'data' array within a signal dictionary to a specified target shape using `numpy.reshape`. The total number of elements must remain the same after reshaping.

**Parameters**

- **shape** (`Tuple[int, ...]`): The target shape to which the data should be reshaped. One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
- 

- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be reshaped. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with reshaped 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Reshape are imported, and numpy as np.
# from tyee.dataset.transform import Reshape
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'flat_data': {
        'data': np.arange(12), # A flat array of 12 elements
        'channels': ['MixedData'],
        'freq': None
    }
}

# 2. Instantiate the Reshape transform to reshape data to (3, 4)
reshaper = Reshape(shape=(3, 4), source='flat_data', target='reshaped_data')

# 3. Apply the transform
processed_results = reshaper(results)

# 4. 'processed_results' will now contain 'reshaped_data'.
#    The 'data' will be reshaped from (12,) to (3, 4).
# Example content of processed_results['reshaped_data']:
# {
#   'data': np.array([[ 0,  1,  2,  3],
#                      [ 4,  5,  6,  7],
#                      [ 8,  9, 10, 11]]),
#   'channels': ['MixedData'],
#   'freq': None
# }
# print(processed_results['reshaped_data']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Transpose

~~~python
class Transpose(BaseTransform):
    def __init__(self, axes=None, source: Optional[str] = None, target: Optional[str] = None):
~~~

Transposes the 'data' array within a signal dictionary by permuting its axes according to the `axes` argument, using `numpy.transpose`. If `axes` is `None`, it reverses the order of the axes.

**Parameters**

- **axes** (`tuple of ints, optional`): A tuple or list of integers, a permutation of `[0, 1, ..., N-1]` where N is the number of axes of the data. The i'th axis of the returned array will correspond to the axis numbered `axes[i]` of the input. If `None` (default), the axes are reversed (e.g., for a 2D array, this is the standard transpose).
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be transposed. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with transposed 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Transpose are imported, and numpy as np.
# from tyee.dataset.transform import Transpose
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'original_matrix': {
        'data': np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), # Shape (2, 2, 2)
        'info': 'A 3D array'
    }
}

# 2. Instantiate Transpose to swap axis 0 and axis 2
transposer = Transpose(axes=(2, 1, 0), source='original_matrix', target='transposed_matrix')

# 3. Apply the transform
processed_results = transposer(results)
# Original data shape: (2, 2, 2)
# Transposed data shape with axes=(2,1,0): (2, 2, 2)

# 4. Instantiate Transpose for a full reversal (default behavior if axes=None)
full_reverser = Transpose(source='original_matrix', target='reversed_axes_matrix')
processed_reversed = full_reverser(results.copy())


# print("Transposed (2,1,0) data:\n", processed_results['transposed_matrix']['data'])
# print("Shape:", processed_results['transposed_matrix']['data'].shape)
# print("Reversed axes data:\n", processed_reversed['reversed_axes_matrix']['data'])
# print("Shape:", processed_reversed['reversed_axes_matrix']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## Squeeze

~~~python
class Squeeze(BaseTransform):
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
~~~

Removes single-dimensional entries from the shape of the 'data' array within a signal dictionary, using `numpy.squeeze`.

**Parameters**

- **axis** (`Optional[int]`): Selects a subset of the single-dimensional entries to remove. If an axis is selected with shape entry greater than one, an error is raised. If `None` (default), all single-dimensional entries are removed.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be squeezed. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with squeezed 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Squeeze are imported, and numpy as np.
# from tyee.dataset.transform import Squeeze
# import numpy as np

# 1. Prepare an example 'results' dictionary with data having a singleton dimension
results = {
    'extra_dim_data': {
        'data': np.array([[[1, 2, 3]]]), # Shape (1, 1, 3)
        'info': 'Data with singleton dimensions'
    }
}

# 2. Instantiate Squeeze to remove all singleton dimensions
squeezer_all = Squeeze(source='extra_dim_data', target='squeezed_all_data')
processed_all = squeezer_all(results)
# processed_all['squeezed_all_data']['data'] will have shape (3,)

# 3. Instantiate Squeeze to remove a specific singleton dimension (e.g., axis 0)
squeezer_axis0 = Squeeze(axis=0, source='extra_dim_data', target='squeezed_axis0_data')
processed_axis0 = squeezer_axis0(results.copy())
# processed_axis0['squeezed_axis0_data']['data'] will have shape (1, 3)

# print("Squeezed all singleton dimensions:\n", processed_all['squeezed_all_data']['data'])
# print("Shape:", processed_all['squeezed_all_data']['data'].shape)
# print("Squeezed axis 0:\n", processed_axis0['squeezed_axis0_data']['data'])
# print("Shape:", processed_axis0['squeezed_axis0_data']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## ExpandDims

~~~python
class ExpandDims(BaseTransform):
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
~~~

Expands the shape of the 'data' array within a signal dictionary by inserting a new axis that will appear at the `axis` position in the expanded array shape, using `numpy.expand_dims`.

**Parameters**

- **axis** (`Optional[int]`): Position in the expanded axes where the new axis (or axes) is placed.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') whose dimensions are to be expanded. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with expanded 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and ExpandDims are imported, and numpy as np.
# from tyee.dataset.transform import ExpandDims
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'array_2d': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # Shape (2, 3)
        'info': 'A 2D array'
    }
}

# 2. Instantiate ExpandDims to add a new axis at the beginning (axis=0)
expander_axis0 = ExpandDims(axis=0, source='array_2d', target='expanded_axis0_data')
processed_axis0 = expander_axis0(results)
# processed_axis0['expanded_axis0_data']['data'] will have shape (1, 2, 3)

# 3. Instantiate ExpandDims to add a new axis at the end (axis=-1 or axis=2 for original 2D)
expander_axis_end = ExpandDims(axis=-1, source='array_2d', target='expanded_axis_end_data')
processed_axis_end = expander_axis_end(results.copy())
# processed_axis_end['expanded_axis_end_data']['data'] will have shape (2, 3, 1)

# print("Expanded at axis 0:\n", processed_axis0['expanded_axis0_data']['data'])
# print("Shape:", processed_axis0['expanded_axis0_data']['data'].shape)
# print("Expanded at axis -1:\n", processed_axis_end['expanded_axis_end_data']['data'])
# print("Shape:", processed_axis_end['expanded_axis_end_data']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## Insert

~~~python
class Insert(BaseTransform):
    def __init__(self, indices: Union[int, List[int], np.ndarray], value: Union[int, float] = 0, axis: int = 1, source: Optional[str] = None, target: Optional[str] = None):
~~~

Inserts a specified `value` into the 'data' array within a signal dictionary at given `indices` along a specified `axis`, using `numpy.insert`.

**Parameters**

- **indices** (`Union[int, List[int], np.ndarray]`): Object into which values are inserted. If `axis` is specified, `indices` must be integer or a list/array of integers. If `axis` is `None`, `indices` is a 1-D array_like object that specifies where in the flattened `data` array the `value` should be inserted.
- **value** (`Union[int, float]`): Value(s) to insert. If the type of `value` is different from that of `data`, `value` is converted to the type of `data`. The `value` is broadcast to the correct shape if necessary. Defaults to `0`.
- **axis** (`int`): Axis along which to insert `value`. If `None`, then `data` is flattened before insertion. Defaults to `1`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') into which values will be inserted. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with inserted values in 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Insert are imported, and numpy as np.
# from tyee.dataset.transform import Insert
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'original_array': {
        'data': np.array([[1, 2], [3, 4], [5, 6]]), # Shape (3, 2)
        'info': 'Original array'
    }
}

# 2. Instantiate Insert to insert the value -99 at index 1 along axis 1 (columns)
inserter_cols = Insert(indices=1, value=-99, axis=1, source='original_array', target='inserted_cols_array')
processed_cols = inserter_cols(results)
# processed_cols['inserted_cols_array']['data'] will be:
# np.array([[  1, -99,   2],
#           [  3, -99,   4],
#           [  5, -99,   6]])

# 3. Instantiate Insert to insert the value -77 at indices [0, 2] along axis 0 (rows)
inserter_rows = Insert(indices=[0, 2], value=-77, axis=0, source='original_array', target='inserted_rows_array')
processed_rows = inserter_rows(results.copy())
# processed_rows['inserted_rows_array']['data'] will be (inserting -77 at row 0, then at new row 2):
# np.array([[-77, -77],
#           [  1,   2],
#           [-77, -77],
#           [  3,   4],
#           [  5,   6]])
# Note: np.insert inserts before the specified index. Multiple insertions shift subsequent indices.

# print("Inserted into columns (axis=1):\n", processed_cols['inserted_cols_array']['data'])
# print("Inserted into rows (axis=0):\n", processed_rows['inserted_rows_array']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ImageResize

~~~python
class ImageResize(BaseTransform):
    def __init__(self, size: Tuple[int, int], source: Optional[str] = None, target: Optional[str] = None):
~~~

Resizes the image data (expected in the 'data' field of a signal dictionary) to a specified `size` using `torchvision.transforms.Resize`. The input data is converted to a PyTorch tensor if it's a NumPy array. The output is converted back to a NumPy array. This transform is suitable for image data, typically with shape (C, H, W) or (H, W, C) for NumPy arrays, or PIL Images.

**Parameters**

- **size** (`Tuple[int, int]`): The desired output size as `(height, width)`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data' as an image or image-like array) to be resized. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with resized image 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ImageResize are imported.
# Also numpy as np, torch, torchvision.transforms.Resize, and PIL.Image.
# from tyee.dataset.transform import ImageResize
# import numpy as np
# import torch
# from torchvision.transforms import Resize
# from PIL import Image # Needed for one of the conversion paths

# 1. Prepare an example 'results' dictionary with image-like data
#    Example: A (Channels, Height, Width) NumPy array
results_numpy = {
    'raw_image_np': {
        'data': np.random.randint(0, 256, size=(3, 100, 150), dtype=np.uint8), # C, H, W
        'info': 'NumPy image array'
    }
}
#    Example: A PIL Image (if your pipeline might produce/consume these)
# try:
#     pil_image = Image.fromarray(np.random.randint(0, 256, size=(60, 80, 3), dtype=np.uint8)) # H, W, C for PIL
#     results_pil = {
#         'raw_image_pil': {
#             'data': pil_image,
#             'info': 'PIL image'
#         }
#     }
# except ImportError:
#     results_pil = None # PIL might not be installed

# 2. Instantiate the ImageResize transform to resize to (32, 32)
resizer = ImageResize(size=(32, 32), source='raw_image_np', target='resized_image_np')

# 3. Apply the transform to NumPy array data
processed_numpy = resizer(results_numpy)
# processed_numpy['resized_image_np']['data'] will be a NumPy array of shape (e.g., 3, 32, 32)
# The channel dimension handling depends on torchvision.transforms.Resize behavior with tensors.
# If input was (C,H,W) tensor, output is (C, new_H, new_W) tensor, then (C, new_H, new_W) numpy array.

# print("Resized NumPy image data shape:", processed_numpy['resized_image_np']['data'].shape)

# if results_pil:
#     resizer_pil = ImageResize(size=(40, 50), source='raw_image_pil', target='resized_image_pil')
#     processed_pil = resizer_pil(results_pil)
#     # processed_pil['resized_image_pil']['data'] will be a NumPy array of shape (e.g., 40, 50, 3) or (3, 40, 50)
#     print("Resized PIL image data shape:", processed_pil['resized_image_pil']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## Scale

~~~python
class Scale(BaseTransform):
    def __init__(self, scale_factor: float = 1.0, source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies numerical scaling to the 'data' field within a signal dictionary by multiplying it with a specified `scale_factor`.

**Parameters**

- **scale_factor** (`float`): The factor by which to scale the signal data. Defaults to `1.0`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be scaled. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with scaled 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Scale are imported, and numpy as np.
# from tyee.dataset.transform import Scale
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'original_signal': {
        'data': np.array([1.0, 2.5, -3.0, 4.2]),
        'channels': ['CH_A'],
        'freq': 100.0
    }
}

# 2. Instantiate the Scale transform to multiply data by 2.0
scaler = Scale(scale_factor=2.0, source='original_signal', target='scaled_signal')

# 3. Apply the transform
processed_results = scaler(results)

# 4. 'processed_results' will now contain 'scaled_signal'.
#    The 'data' in 'scaled_signal' will be [2.0, 5.0, -6.0, 8.4].
# Example content of processed_results['scaled_signal']:
# {
#   'data': np.array([2.0, 5.0, -6.0, 8.4]),
#   'channels': ['CH_A'],
#   'freq': 100.0
# }
# print(processed_results['scaled_signal'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Offset

~~~python
class Offset(BaseTransform):
    def __init__(self, offset: float | int = 0.0, source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a numerical offset to the 'data' field within a signal dictionary by adding a specified `offset` value to it.

**Parameters**

- **offset** (`float | int`): The value to add to the signal data. Defaults to `0.0`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be offset. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with offset 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Offset are imported, and numpy as np.
# from tyee.dataset.transform import Offset
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'signal_to_offset': {
        'data': np.array([10, 20, 30, 40]),
        'channels': ['SensorX'],
        'freq': 50.0
    }
}

# 2. Instantiate the Offset transform to add 5 to the data
offset_adder = Offset(offset=5.0, source='signal_to_offset', target='offset_signal')

# 3. Apply the transform
processed_results = offset_adder(results)

# 4. 'processed_results' will now contain 'offset_signal'.
#    The 'data' in 'offset_signal' will be [15, 25, 35, 45].
# Example content of processed_results['offset_signal']:
# {
#   'data': np.array([15, 25, 35, 45]),
#   'channels': ['SensorX'],
#   'freq': 50.0
# }
# print(processed_results['offset_signal'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Round

~~~python
class Round(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
~~~

Rounds the numerical values in the 'data' field of a signal dictionary to the nearest integer, using `numpy.round`.

**Parameters**

- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be rounded. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with rounded 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Round are imported, and numpy as np.
# from tyee.dataset.transform import Round
# import numpy as np

# 1. Prepare an example 'results' dictionary with floating point data
results = {
    'float_signal': {
        'data': np.array([1.2, 2.7, 3.5, 4.9, -0.3]),
        'channels': ['Values'],
    }
}

# 2. Instantiate the Round transform
rounder = Round(source='float_signal', target='rounded_signal')

# 3. Apply the transform
processed_results = rounder(results)

# 4. 'processed_results' will now contain 'rounded_signal'.
#    The 'data' in 'rounded_signal' will be [1., 3., 4., 5., -0.].
# Example content of processed_results['rounded_signal']:
# {
#   'data': np.array([1., 3., 4., 5., -0.]),
#   'channels': ['Values']
# }
# print(processed_results['rounded_signal'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Log

~~~python
class Log(BaseTransform):
    def __init__(self, epsilon:float=1e-10, source: Optional[str] = None, target: Optional[str] = None):
~~~

Applies a natural logarithm transformation to the 'data' field within a signal dictionary, using `numpy.log`. An epsilon value is added to the data before taking the logarithm to avoid issues with log(0).

**Parameters**

- **epsilon** (`float`): A small constant added to the data before applying the logarithm to prevent `log(0)` errors. Defaults to `1e-10`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be transformed. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with log-transformed 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Log are imported, and numpy as np.
# from tyee.dataset.transform import Log
# import numpy as np

# 1. Prepare an example 'results' dictionary with positive data
results = {
    'positive_signal': {
        'data': np.array([1, 10, 100, 0.1, 0.00000000001]), # Includes a very small positive number
        'channels': ['Intensity'],
    }
}

# 2. Instantiate the Log transform
log_transformer = Log(epsilon=1e-10, source='positive_signal', target='log_signal')

# 3. Apply the transform
processed_results = log_transformer(results)

# 4. 'processed_results' will now contain 'log_signal'.
#    The 'data' in 'log_signal' will be the natural logarithm of the original data (plus epsilon).
#    e.g., np.log(1 + 1e-10) is approx 0.
#    np.log(10 + 1e-10) is approx 2.302585.
#    np.log(0.00000000001 + 1e-10) is approx np.log(1.1e-10) approx -22.92
# print(processed_results['log_signal'])
~~~

[`Back to Top`](#tyeedatasettransform)

## Select

~~~python
class Select(BaseTransform):
    def __init__(self, key: Union[str, List[str]], source: Optional[str] = None, target: Optional[str] = None):
~~~

Selects a subset of entries from an input dictionary (which is typically a signal dictionary) based on a specified key or list of keys. The `source` and `target` parameters from `BaseTransform` are not directly used by this transform's core logic, as it operates on the dictionary passed to its `transform` method. When used within a `BaseTransform` pipeline, `source` would dictate which dictionary in a larger results set is passed to `transform`, and `target` where the new, filtered dictionary is placed.

**Parameters**

- **key** (`Union[str, List[str]]`): A single key (string) or a list of keys (strings) to be retained in the output dictionary.
- **source** (`Optional[str]`): The key identifying the input dictionary within a larger results structure if applicable. Not directly used by the selection logic itself but by the `BaseTransform` pipeline. Defaults to `None`.
- **target** (`Optional[str]`): The key under which the new dictionary (with selected keys) will be stored in a larger results structure if applicable. Not directly used by the selection logic itself but by the `BaseTransform` pipeline. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and Select are imported.
# from tyee.dataset.transform import Select

# 1. Prepare an example 'results' dictionary (simulating a signal dictionary)
signal_dict = {
    'data': [1, 2, 3, 4, 5],
    'freq': 100.0,
    'channels': ['CH1', 'CH2'],
    'status': 'processed',
    'subject_id': 'S001'
}

# Example of how it might be used in a BaseTransform pipeline:
results_pipeline_context = {
    'raw_signal_info': signal_dict
}


# 2. Instantiate Select to keep only 'data' and 'freq'
selector_data_freq = Select(key=['data', 'freq'], source='raw_signal_info', target='selected_info')

# 3. Apply the transform (simulating how BaseTransform.__call__ would use it)
#    In a real pipeline, you'd call selector_data_freq(results_pipeline_context)
#    Here, we simulate the direct call to its transform method for clarity on what Select does:
#    selected_dictionary = selector_data_freq.transform(results_pipeline_context['raw_signal_info'])
#    For a direct call:
selected_dictionary = selector_data_freq.transform(signal_dict)


# 4. 'selected_dictionary' will contain only the 'data' and 'freq' keys.
# Expected content of selected_dictionary:
# {
#   'data': [1, 2, 3, 4, 5],
#   'freq': 100.0
# }
# print(selected_dictionary)

# Example selecting a single key
selector_subject = Select(key='subject_id')
selected_subject_dict = selector_subject.transform(signal_dict)
# Expected content of selected_subject_dict:
# {
#   'subject_id': 'S001'
# }
# print(selected_subject_dict)
~~~

[`Back to Top`](#tyeedatasettransform)

## SlideWindow

~~~python
class SlideWindow(BaseTransform):
    def __init__(
        self,
        window_size: int,
        stride: int,
        axis: int = -1,
        source: Optional[str] = None,
        target: Optional[str] = None,
        keep_tail: bool = False
    ):
~~~

Calculates the start and end indices for sliding windows along a specified axis of the 'data' in a signal dictionary. This transform does not modify the 'data' itself but adds a list of window indices to `result['info']['windows']` and stores the `axis` used in `result['axis']`. These outputs are typically consumed by a subsequent transform like `WindowExtract`.

**Parameters**

- **window_size** (`int`): The length of each sliding window.
- **stride** (`int`): The step size or overlap between consecutive windows.
- **axis** (`int`): The axis along which to slide the windows (typically the time axis). Defaults to `-1`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data') to be windowed. Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the signal dictionary (now including windowing information) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.
- **keep_tail** (`bool`): If `True` and the last window does not fully fit with the given stride, a final window of `window_size` ending at the very end of the data will be included. This might overlap more than specified by `stride` with the preceding window. Defaults to `False`.

**Usage Example**

~~~python
# Assume BaseTransform and SlideWindow are imported, and numpy as np.
# from tyee.dataset.transform import SlideWindow
# import numpy as np

# 1. Prepare an example 'results' dictionary
results = {
    'continuous_signal': {
        'data': np.arange(10).reshape(1, 10), # 1 channel, 10 time points
        'channels': ['CH1'],
        'freq': 100.0
    }
}

# 2. Instantiate SlideWindow to create windows of size 5 with a stride of 2
#    along the last axis (time).
window_definer = SlideWindow(
    window_size=5,
    stride=2,
    axis=-1,
    keep_tail=False,
    source='continuous_signal',
    target='windowed_signal_info'
)

# 3. Apply the transform
processed_results = window_definer(results)

# 4. 'processed_results' will now contain 'windowed_signal_info'.
#    - 'windowed_signal_info['data']' remains the original data.
#    - 'windowed_signal_info['info']['windows']' will contain the window indices.
#    - 'windowed_signal_info['axis']' will be -1.
#
# Expected 'info']['windows'] for window_size=5, stride=2, length=10, keep_tail=False:
# [{'start': 0, 'end': 5}, {'start': 2, 'end': 7}, {'start': 4, 'end': 9}]
# (The window starting at 6 would be [6,7,8,9,10], but data length is 10, so end is 11, which is out of bounds if not for keep_tail.
#  If length was 11, start=6, end=11 would be valid.
#  With length 10, the last full window is start=4, end=9.
#  If keep_tail=True, and (10-5)%2 != 0 (which is 5%2=1, true), it would add:
#  {'start': 10-5=5, 'end': 10} -> [{'start': 0, 'end': 5}, {'start': 2, 'end': 7}, {'start': 4, 'end': 9}, {'start': 5, 'end': 10}]

# print(processed_results['windowed_signal_info']['info']['windows'])
# print(processed_results['windowed_signal_info']['axis'])
~~~

[`Back to Top`](#tyeedatasettransform)

## WindowExtract

~~~python
class WindowExtract(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
~~~

Extracts data segments (windows) from the 'data' array in a signal dictionary based on the start/end indices provided in `result['info']['windows']` (typically generated by `SlideWindow`). The extracted windows are then stacked along a new first dimension, resulting in an output 'data' array of shape (num_windows, channels, window_size) or similar, depending on the original data dimensions and the windowing axis.

**Parameters**

- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary. This dictionary should contain 'data', and `info['windows']` (from `SlideWindow`). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the new signal dictionary (with 'data' now being the stacked windows) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, SlideWindow, WindowExtract are imported, and numpy as np.
# from tyee.dataset.transform import SlideWindow, WindowExtract
# import numpy as np

# 1. Prepare initial data and define windows using SlideWindow
initial_results = {
    'continuous_signal': {
        'data': np.arange(20).reshape(2, 10), # 2 channels, 10 time points each
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}
window_definer = SlideWindow(
    window_size=4,
    stride=2,
    axis=-1, # Window along the time axis
    source='continuous_signal',
    target='signal_with_windows' # SlideWindow modifies this entry
)
results_with_window_info = window_definer(initial_results)
# results_with_window_info['signal_with_windows']['info']['windows'] will be:
# [{'start': 0, 'end': 4}, {'start': 2, 'end': 6}, {'start': 4, 'end': 8}, {'start': 6, 'end': 10}]

# 2. Instantiate WindowExtract
#    'source' should point to the key modified by SlideWindow.
window_extractor = WindowExtract(source='signal_with_windows', target='extracted_windows_data')

# 3. Apply the WindowExtract transform
processed_results = window_extractor(results_with_window_info)

# 4. 'processed_results' will now contain 'extracted_windows_data'.
#    The 'data' in 'extracted_windows_data' will be a stacked array of the windows.
#    Original data for CH1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#    Windows for CH1: [0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]
#    Original data for CH2: [10,11,12,13,14,15,16,17,18,19]
#    Windows for CH2: [10,11,12,13], [12,13,14,15], [14,15,16,17], [16,17,18,19]
#    Stacked 'data' shape: (num_windows, num_channels, window_size) -> (4, 2, 4)
#
# Example content of processed_results['extracted_windows_data']['data']:
# First window (all channels): [[0,1,2,3], [10,11,12,13]]
# Second window (all channels): [[2,3,4,5], [12,13,14,15]]
# ...etc.
# print(processed_results['extracted_windows_data']['data'])
# print(processed_results['extracted_windows_data']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## CWTSpectrum

~~~python
class CWTSpectrum(BaseTransform):
    def __init__(self, freqs, output_type='power', n_jobs=1, verbose=0, source=None, target=None):
~~~

Computes the time-frequency representation of the input signal 'data' using Continuous Wavelet Transform (CWT) with Morlet wavelets, leveraging `mne.time_frequency.tfr_array_morlet`. The input data is expected to be 2D (channels, time).

**Parameters**

- **freqs** (`array-like of float`): The frequencies of interest for the CWT.
- **output_type** (`str`): The type of output to compute. Can be 'power', 'phase', 'avg_power_itc', 'itc', or 'complex'. Defaults to `'power'`.
- **n_jobs** (`int`): The number of jobs to run in parallel. Defaults to `1`.
- **verbose** (`int | bool | str | None`): Controls verbosity. See `mne.time_frequency.tfr_array_morlet` for details. Defaults to `0`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data' and 'freq'). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with CWT spectrum as 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and CWTSpectrum are imported, and numpy as np, mne.
# from tyee.dataset.transform import CWTSpectrum
# import numpy as np
# import mne # For mne.time_frequency.tfr_array_morlet

# 1. Prepare an example 'results' dictionary
results = {
    'eeg_signal': {
        'data': np.random.randn(3, 1000), # 3 channels, 1000 time points
        'freq': 250.0, # Sampling frequency
        'channels': ['CH1', 'CH2', 'CH3']
    }
}

# 2. Define frequencies of interest for CWT
cwt_frequencies = np.arange(5, 30, 2) # Frequencies from 5 Hz to 28 Hz, step 2 Hz

# 3. Instantiate the CWTSpectrum transform
cwt_transformer = CWTSpectrum(
    freqs=cwt_frequencies,
    output_type='power',
    source='eeg_signal',
    target='cwt_spectrum_output'
)

# 4. Apply the transform
processed_results = cwt_transformer(results)

# 5. 'processed_results' will now contain 'cwt_spectrum_output'.
#    The 'data' in 'cwt_spectrum_output' will be the CWT power spectrum,
#    with shape (num_channels, num_cwt_freqs, num_time_points).
#    e.g., (3, len(cwt_frequencies), 1000)
# print(processed_results['cwt_spectrum_output']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## DWTSpectrum

~~~python
class DWTSpectrum(BaseTransform):
    def __init__(self, wavelet='db4', level=4, source=None, target=None):
~~~

Computes the Discrete Wavelet Transform (DWT) coefficients for each channel of the input signal 'data' using `pywt.wavedec`. The coefficients from different decomposition levels for each channel are concatenated.

**Parameters**

- **wavelet** (`str`): The name of the wavelet to use (e.g., 'db4', 'haar', 'sym5'). Defaults to `'db4'`.
- **level** (`int`): The decomposition level. Defaults to `4`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data'). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with DWT coefficients as 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and DWTSpectrum are imported, and numpy as np, pywt.
# from tyee.dataset.transform import DWTSpectrum
# import numpy as np
# import pywt # For pywt.wavedec

# 1. Prepare an example 'results' dictionary
results = {
    'time_series': {
        'data': np.random.randn(2, 512), # 2 channels, 512 time points
        'freq': 128.0,
        'channels': ['SensorA', 'SensorB']
    }
}

# 2. Instantiate the DWTSpectrum transform
dwt_transformer = DWTSpectrum(
    wavelet='db4',
    level=3,
    source='time_series',
    target='dwt_coeffs_output'
)

# 3. Apply the transform
processed_results = dwt_transformer(results)

# 4. 'processed_results' will now contain 'dwt_coeffs_output'.
#    The 'data' in 'dwt_coeffs_output' will be an array where each row
#    contains the concatenated DWT coefficients for the corresponding input channel.
#    The length of concatenated coefficients depends on the original signal length, wavelet, and level.
# print(processed_results['dwt_coeffs_output']['data'].shape)
~~~

[`Back to Top`](#tyeedatasettransform)

## FFTSpectrum

~~~python
class FFTSpectrum(BaseTransform):
    def __init__(
        self,
        resolution: Optional[int] = None,
        min_hz: Optional[float] = None,
        max_hz: Optional[float] = None,
        axis: int = 0, # Note: FFT is typically applied on the time axis, often the last axis. Defaulting to 0 might be unusual depending on data layout.
        sample_rate_key: str = 'freq',
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

Computes the Fast Fourier Transform (FFT) magnitude spectrum of the 'data' in a signal dictionary using `scipy.fft.rfft`. It supports padding or truncating the signal to a specified `resolution` along the transform `axis`, and filtering the resulting spectrum to a frequency range defined by `min_hz` and `max_hz`. The output 'data' will be the magnitude spectrum, and a 'freqs' key with corresponding frequencies will be added.

**Parameters**

- **resolution** (`Optional[int]`): The desired length of the signal along the `axis` for FFT computation. If the signal is shorter, it's zero-padded; if longer, it's truncated. If `None`, the original length is used. Defaults to `None`.
- **min_hz** (`Optional[float]`): The minimum frequency to include in the output spectrum. If `None`, no lower bound is applied. Defaults to `None`.
- **max_hz** (`Optional[float]`): The maximum frequency to include in the output spectrum. If `None`, no upper bound is applied. Defaults to `None`.
- **axis** (`int`): The axis of the input 'data' along which the FFT is computed (typically the time axis). Defaults to `0`. *User should ensure this is appropriate for their data's (channel, time) or (time, channel) convention.*
- **sample_rate_key** (`str`): The key in the input signal dictionary that holds the sampling rate value. Defaults to `'freq'`.
- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and FFTSpectrum are imported, and numpy as np, scipy.fft.
# from tyee.dataset.transform import FFTSpectrum
# import numpy as np
# from scipy.fft import rfft, rfftfreq # For the class to function

# 1. Prepare example data
sampling_freq = 200.0
time_points = 400 # 2 seconds of data
data_array = np.random.randn(3, time_points) # 3 channels, 400 time points
results = {
    'eeg_data': {
        'data': data_array,
        'freq': sampling_freq, # Correct key for sample_rate_key
        'channels': ['C3', 'C4', 'Cz']
    }
}

# 2. Instantiate FFTSpectrum to get spectrum from 1 Hz to 30 Hz, with a resolution of 512 points for FFT
#    Assuming time is the last axis, so axis should be -1 or 1 for (channels, time)
fft_transformer = FFTSpectrum(
    resolution=512,
    min_hz=1.0,
    max_hz=30.0,
    axis=-1, # Apply FFT along the last axis (time)
    sample_rate_key='freq',
    source='eeg_data',
    target='fft_output'
)

# 3. Apply the transform
processed_results = fft_transformer(results)

# 4. 'processed_results' will contain 'fft_output'.
#    'fft_output['data']' will be the magnitude spectrum for the specified frequency band.
#    'fft_output['freqs']' will be the corresponding frequency bins.
# print("FFT Spectrum data shape:", processed_results['fft_output']['data'].shape)
# print("Frequency bins:", processed_results['fft_output']['freqs'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ToImage

~~~python
class ToImage(BaseTransform):
    def __init__(self, length: int, width: int, resize_length_factor: float, native_resnet_size: int,
                 cmap: str = 'viridis', source: str = None, target: str = None):
~~~

Converts input signal data (typically 2D, e.g., channels x time) into an image representation. The process involves contrast normalization, applying a colormap, reshaping, resizing using `torchvision.transforms.Resize`, another contrast normalization after interpolation, and finally ImageNet-style normalization. The output is a NumPy array of type `np.float32`.

**Parameters**

- **length** (`int`): Target length for the image after resizing (this is multiplied by `resize_length_factor`).
- **width** (`int`): The width to which the input data is initially reshaped (per channel, before colormapping). This often corresponds to the number of time points or features per channel segment.
- **resize_length_factor** (`float`): A factor by which `length` is multiplied to determine the final resized image length.
- **native_resnet_size** (`int`): The target size for the other dimension of the image after resizing (typically corresponding to a ResNet input dimension).
- **cmap** (`str`): The name of the Matplotlib colormap to apply to the data. Defaults to `'viridis'`.
- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data' and 'channels'). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with image as 'data') will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and ToImage are imported.
# Also numpy as np, torch, matplotlib as mpl, torchvision.transforms.
# from tyee.dataset.transform import ToImage
# import numpy as np
# import torch
# import matplotlib as mpl
# from torchvision import transforms

# 1. Prepare example data (e.g., 2 channels, 20 time points)
results = {
    'eeg_segment': {
        'data': np.random.rand(2, 20), 
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. Instantiate ToImage
#    Parameters are illustrative; native_resnet_size often 224 for ResNet.
#    Width should match the number of time points in the input data (20 in this case).
image_converter = ToImage(
    length=64,              # Target length for one dimension of the image
    width=20,               # Should match the width of the input data (e.g., time points)
    resize_length_factor=1.0, # Factor for final length resizing
    native_resnet_size=64,  # Target for the other dimension (e.g., ResNet's expected input size)
    cmap='jet',
    source='eeg_segment',
    target='image_representation'
)

# 3. Apply the transform
processed_results = image_converter(results)

# 4. 'processed_results' will now contain 'image_representation'.
#    The 'data' will be a NumPy float32 array representing the image.
#    Shape would be (3, length * resize_length_factor, native_resnet_size) e.g. (3, 64, 64)
# print(processed_results['image_representation']['data'].shape)
# print(processed_results['image_representation']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToTensor

~~~python
class ToTensor(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field (expected to be a NumPy array) within a signal dictionary to a PyTorch tensor using `torch.from_numpy()`. The data type of the resulting tensor is inferred from the NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data'). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with 'data' as a PyTorch tensor) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and ToTensor are imported, and numpy as np, torch.
# from tyee.dataset.transform import ToTensor
# import numpy as np
# import torch

# 1. Prepare an example 'results' dictionary with NumPy data
results = {
    'numpy_data_signal': {
        'data': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'info': 'Some numeric data'
    }
}

# 2. Instantiate the ToTensor transform
to_tensor_converter = ToTensor(source='numpy_data_signal', target='tensor_data_signal')

# 3. Apply the transform
processed_results = to_tensor_converter(results)

# 4. 'processed_results' will now contain 'tensor_data_signal'.
#    The 'data' in 'tensor_data_signal' will be a PyTorch tensor.
#    The dtype will be torch.int32 in this case.
# print(isinstance(processed_results['tensor_data_signal']['data'], torch.Tensor))
# print(processed_results['tensor_data_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToTensorFloat32

~~~python
class ToTensorFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a PyTorch tensor and then casts it to `float32` (torch.float). If the input 'data' is already a PyTorch tensor, it's directly cast to `float32`. If it's a NumPy array, it's first converted to a tensor.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToTensorFloat32, numpy as np, torch.
# from tyee.dataset.transform import ToTensorFloat32
# import numpy as np
# import torch

results = {
    'int_numpy_signal': {'data': np.array([1, 2, 3], dtype=np.int64)}
}
converter = ToTensorFloat32(source='int_numpy_signal', target='float32_tensor_signal')
processed = converter(results)
# processed['float32_tensor_signal']['data'] is a torch.float32 tensor
# print(processed['float32_tensor_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToTensorFloat16

~~~python
class ToTensorFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a PyTorch tensor and then casts it to `float16` (torch.half). If the input 'data' is already a PyTorch tensor, it's directly cast to `float16`. If it's a NumPy array, it's first converted to a tensor.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToTensorFloat16, numpy as np, torch.
# from tyee.dataset.transform import ToTensorFloat16
# import numpy as np
# import torch

results = {
    'float_numpy_signal': {'data': np.array([1.0, 2.0, 3.0], dtype=np.float64)}
}
converter = ToTensorFloat16(source='float_numpy_signal', target='float16_tensor_signal')
processed = converter(results)
# processed['float16_tensor_signal']['data'] is a torch.float16 tensor
# print(processed['float16_tensor_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToTensorInt64

~~~python
class ToTensorInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a PyTorch tensor and then casts it to `int64` (torch.long). If the input 'data' is already a PyTorch tensor, it's directly cast to `int64`. If it's a NumPy array, it's first converted to a tensor.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToTensorInt64, numpy as np, torch.
# from tyee.dataset.transform import ToTensorInt64
# import numpy as np
# import torch

results = {
    'float_numpy_signal': {'data': np.array([1.0, 2.7, 3.1], dtype=np.float32)}
}
converter = ToTensorInt64(source='float_numpy_signal', target='int64_tensor_signal')
processed = converter(results)
# processed['int64_tensor_signal']['data'] is a torch.int64 tensor (values will be truncated: [1, 2, 3])
# print(processed['int64_tensor_signal']['data'].dtype)
# print(processed['int64_tensor_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpy

~~~python
class ToNumpy(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field (expected to be a PyTorch tensor) within a signal dictionary to a NumPy array using the tensor's `.numpy()` method.

**Parameters**

- **source** (`Optional[str]`): The key in the input dictionary that holds the signal dictionary (containing 'data' as a PyTorch tensor). Defaults to `None`.
- **target** (`Optional[str]`): The key in the output dictionary where the transformed signal dictionary (with 'data' as a NumPy array) will be stored. If `None`, it defaults to the `source` key. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform and ToNumpy are imported, and numpy as np, torch.
# from tyee.dataset.transform import ToNumpy
# import numpy as np
# import torch

# 1. Prepare an example 'results' dictionary with PyTorch tensor data
results = {
    'tensor_data_signal': {
        'data': torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
        'info': 'Some tensor data'
    }
}

# 2. Instantiate the ToNumpy transform
to_numpy_converter = ToNumpy(source='tensor_data_signal', target='numpy_data_signal')

# 3. Apply the transform
processed_results = to_numpy_converter(results)

# 4. 'processed_results' will now contain 'numpy_data_signal'.
#    The 'data' in 'numpy_data_signal' will be a NumPy array.
#    The dtype will be float32 in this case.
# print(isinstance(processed_results['numpy_data_signal']['data'], np.ndarray))
# print(processed_results['numpy_data_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpyFloat64

~~~python
class ToNumpyFloat64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a NumPy array and then casts it to `np.float64`. If the input 'data' is a PyTorch tensor, it's first converted to a NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToNumpyFloat64, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat64
# import numpy as np
# import torch

results = {
    'int_tensor_signal': {'data': torch.tensor([1, 2, 3], dtype=torch.int32)}
}
converter = ToNumpyFloat64(source='int_tensor_signal', target='float64_numpy_signal')
processed = converter(results)
# processed['float64_numpy_signal']['data'] is a np.float64 array
# print(processed['float64_numpy_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpyFloat32

~~~python
class ToNumpyFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a NumPy array and then casts it to `np.float32`. If the input 'data' is a PyTorch tensor, it's first converted to a NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToNumpyFloat32, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat32
# import numpy as np
# import torch

results = {
    'int_tensor_signal': {'data': torch.tensor([1, 2, 3], dtype=torch.int64)}
}
converter = ToNumpyFloat32(source='int_tensor_signal', target='float32_numpy_signal')
processed = converter(results)
# processed['float32_numpy_signal']['data'] is a np.float32 array
# print(processed['float32_numpy_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpyFloat16

~~~python
class ToNumpyFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a NumPy array and then casts it to `np.float16`. If the input 'data' is a PyTorch tensor, it's first converted to a NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToNumpyFloat16, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat16
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
}
converter = ToNumpyFloat16(source='float_tensor_signal', target='float16_numpy_signal')
processed = converter(results)
# processed['float16_numpy_signal']['data'] is a np.float16 array
# print(processed['float16_numpy_signal']['data'].dtype)
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpyInt64

~~~python
class ToNumpyInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a NumPy array and then casts it to `np.int64`. If the input 'data' is a PyTorch tensor, it's first converted to a NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToNumpyInt64, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyInt64
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.1, 2.7, 3.9], dtype=torch.float32)}
}
converter = ToNumpyInt64(source='float_tensor_signal', target='int64_numpy_signal')
processed = converter(results)
# processed['int64_numpy_signal']['data'] is a np.int64 array (values truncated: [1, 2, 3])
# print(processed['int64_numpy_signal']['data'].dtype)
# print(processed['int64_numpy_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)

## ToNumpyInt32

~~~python
class ToNumpyInt32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

Converts the 'data' field within a signal dictionary to a NumPy array and then casts it to `np.int32`. If the input 'data' is a PyTorch tensor, it's first converted to a NumPy array.

**Parameters**

- **source** (`Optional[str]`): The key for the input signal dictionary. Defaults to `None`.
- **target** (`Optional[str]`): The key for the output signal dictionary. Defaults to `None`.

**Usage Example**

~~~python
# Assume BaseTransform, ToNumpyInt32, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyInt32
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.1, 2.7, 3.9], dtype=torch.float64)}
}
converter = ToNumpyInt32(source='float_tensor_signal', target='int32_numpy_signal')
processed = converter(results)
# processed['int32_numpy_signal']['data'] is a np.int32 array (values truncated: [1, 2, 3])
# print(processed['int32_numpy_signal']['data'].dtype)
# print(processed['int32_numpy_signal']['data'])
~~~

[`Back to Top`](#tyeedatasettransform)