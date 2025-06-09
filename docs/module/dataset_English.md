# tyee.dataset
## Table of Contents

- [How to Build a Custom Dataset (Inheriting `BaseDataset`)](#how-to-build-a-custom-dataset)

| Dataset Class Name                                           | Description                                                  | Data Types         | Sampling Rate   | Application Domain |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | --------------- | ------------------ |
| [`BaseDataset`](#basedataset)                               | Base class for all dataset classes, providing core functionalities for dataset initialization, preprocessing, caching, and loading | Generic            | Configurable    | Base Framework     |
| [`BCICIV2ADataset`](#bciciv2adataset)                       | Processes BCI Competition IV Dataset 2a for motor imagery task classification | EEG, EOG           | 250 Hz          | Brain-Computer Interface |
| [`BCICIV4Dataset`](#bciciv4dataset)                         | Processes BCI Competition IV Dataset 4 ECoG data for finger flexion prediction | ECoG, Dataglove    | 1000 Hz         | Brain-Computer Interface |
| [`CinC2018Dataset`](#cinc2018dataset)                       | Processes PhysioNet/CinC Challenge 2018 dataset for automatic sleep stage classification | EEG, EOG, EMG, etc | 200 Hz          | Sleep Analysis     |
| [`DaLiADataset`](#daliadataset)                             | Processes PPG_DaLiA dataset for PPG-based heart rate estimation | PPG, ACC           | 64 Hz, 32 Hz    | Heart Rate Monitoring |
| [`DEAPDataset`](#deapdataset)                               | Processes DEAP dataset for emotion recognition using multimodal physiological signals | EEG, EOG, EMG, etc | 128 Hz          | Affective Computing |
| [`KaggleERNDataset`](#kaggleerndataset)                     | Processes Kaggle "Grasp-and-Lift EEG Detection" dataset for error-related negativity detection | EEG                | 200 Hz          | Brain-Computer Interface |
| [`MITBIHDataset`](#mitbihdataset)                           | Processes MIT-BIH Arrhythmia Database for ECG arrhythmia detection | ECG                | 360 Hz          | Cardiac Analysis   |
| [`NinaproDB5Dataset`](#ninaprodb5dataset)                   | Processes Ninapro DB5 dataset for sEMG-based gesture recognition | sEMG               | 200 Hz          | Gesture Recognition |
| [`PhysioP300Dataset`](#physiop300dataset)                   | Processes PhysioNet P300 dataset for P300 speller and other BCI applications | EEG                | 2048 Hz         | Brain-Computer Interface |
| [`SEEDVFeatureDataset`](#seedvfeaturedataset)               | Processes pre-extracted features from SEED-V dataset for multimodal emotion recognition | EEG Features, Eye Features | Feature Data    | Affective Computing |
| [`SleepEDFCassetteDataset`](#sleepedfcassetedataset)        | Processes Sleep-EDF Database Expanded cassette part for sleep stage analysis | EEG, EOG, EMG, etc | 100 Hz, 1 Hz    | Sleep Analysis     |
| [`SleepEDFTelemetryDataset`](#sleepedftelemetrydataset)     | Processes Sleep-EDF Database Expanded telemetry part for sleep stage analysis | EEG, EOG, EMG      | 100 Hz          | Sleep Analysis     |
| [`TUABDataset`](#tuabdataset)                               | Processes TUAB dataset for clinical EEG abnormality detection | EEG                | 256 Hz          | Clinical Diagnosis |
| [`TUEVDataset`](#tuevdataset)                               | Processes TUH EEG Events dataset for EEG event detection    | EEG                | 256 Hz          | Clinical Diagnosis |


## BaseDataset

```python
class BaseDataset(Dataset):
    def __init__(
        self,
        root_path: str,
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
        **kwargs
    ) -> None:
```

`BaseDataset` is a base class for managing various types of physiological signal datasets. It provides core functionalities such as dataset initialization, raw data processing, data reading and writing, metadata management, switching between lazy/eager loading modes, and applying preprocessing transforms ("hooks"). When the cache path specified by `io_path` is empty or `io_mode` is `'memory'`, data preprocessing is automatically performed, and the results are written to `io_path`; otherwise, data is loaded directly from the cache.

**Core Design Philosophy:**

The design of `BaseDataset` aims to decouple raw data reading, processing, segmentation, storage, and final data retrieval. By inheriting this class and implementing specific abstract methods, developers can quickly build processing workflows for different raw dataset formats. The processed data and metadata are stored efficiently (e.g., using HDF5 or LMDB) for subsequent training and analysis.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path where the original dataset is located.
- **io_path** (`Union[None, str]`): Path for storing/reading preprocessed results (cache). If `None` or if no valid cache is found at the path, the data preprocessing workflow will be triggered.
- **start_offset** (`float`): Offset in seconds applied to the start time of each segment when extracting from the original signal. Defaults to `0.0`.
- **end_offset** (`float`): Offset in seconds applied to the end time of each segment. Defaults to `0.0`.
- **include_end** (`bool`): Whether to include the sample corresponding to the calculated end time (after `end_offset`) when segmenting by time. Defaults to `False` (i.e., exclusive).
- **before_segment_transform** (`Union[None, Callable]`): An optional transform function (or list of transforms) applied to the `signals` part of the full record data read by `read_record`, *before* data segmentation (call to `segment_split`).
- **offline_signal_transform** (`Union[None, Callable]`): An optional transform function (or list of transforms) applied to signal data during the data processing phase (inside `process_record`, typically after each segment is formed). The transformed results are cached.
- **offline_label_transform** (`Union[None, Callable]`): An optional transform function (or list of transforms) applied to label data during the data processing phase. The transformed results are cached.
- **online_signal_transform** (`Union[None, Callable]`): An optional transform function (or list of transforms) dynamically applied to the signal data read during the data loading phase (in `__getitem__`). The transformed results are *not* cached.
- **online_label_transform** (`Union[None, Callable]`): An optional transform function (or list of transforms) dynamically applied to the label data read during the data loading phase. The transformed results are *not* cached.
- **io_mode** (`str`): Storage mode for preprocessed results, e.g., 'hdf5', 'lmdb', 'pickle', 'memory'. Defaults to `'hdf5'`.
- **io_size** (`int`): Maximum capacity of the database in bytes. Defaults to `1048576` (1MB).
- **io_chunks** (`int`, optional): Data chunk size for HDF5 storage mode.
- **num_worker** (`int`): Number of parallel worker processes to use during data preprocessing. `0` means single process. Defaults to `0`.
- **lazy_threshold** (`int`): When the number of records in `io_path` exceeds this threshold, the dataset switches to lazy loading mode (IO handlers are loaded only when needed). Defaults to `128`.
- **verbose** (`bool`): Whether to display progress bars during data preprocessing. Defaults to `True`.
- ***\*kwargs**: Other parameters passed to methods in the concrete dataset implementation (like `set_records`, `read_record`, `process_record`).

[`Back to Top`](#tyeedataset)

## How to Build a Custom Dataset 

To build a processing workflow for a specific raw dataset based on `BaseDataset`, you need to create a new class that inherits from `BaseDataset` and implement at least the following three core abstract methods: `set_records`, `read_record`, and `process_record`.

### 1. Mandatory Methods to Implement

The following methods **must be overridden** in your subclass:

- **`set_records(self, root_path: str, \**kwargs) -> List[Any]`**
  - **Function**: Defines a list of raw data record units to be processed. These record units can be filenames, tuples of file paths, or any identifier that the `read_record` method can understand and use to locate the raw data.
  - **Parameters**:
    - `root_path` (`str`): Typically the root directory where raw data is stored, passed from `BaseDataset`'s `__init__`.
    - `**kwargs`: Other parameters passed from `BaseDataset`'s `__init__`.
  - **Returns**: `List[Any]` - A list where each element represents a record to be processed.
  - **Example**:

```python
def set_records(self, root_path: str, **kwargs) -> List[str]:
    # Assume raw data are .edf files in the root directory
    records = []
    for filename in os.listdir(root_path):
        if filename.endswith('.edf'):
            records.append(os.path.join(root_path, filename))
    return records
```

- `read_record(self, record: Any, \**kwargs) -> Dict`
  - **Function**: Receives a record identifier from the list returned by `set_records` and is responsible for reading all relevant data for that raw record, including all signal types, labels, and metadata.
  - **Parameters**:
    - `record` (`Any`): An element from the list returned by `set_records`.
    - `**kwargs`: Other parameters passed from `BaseDataset`'s `__init__`.
  - **Returns**: `Dict` - A dictionary containing the read data, structured as follows:
    - Top-level key `'signals'`(required): Its value is a dictionary where each key is a signal type name (e.g., 'eeg', 'ecg'), and the corresponding value is another dictionary containing:
      - `'data': np.ndarray`: Signal data (e.g., a `(num_channels, num_samples)` array).
      - `'channels': List[str]`: List of channel names.
      - `'freq': float`: Sampling frequency of this signal.
    - Top-level key `'labels'`(required): Its value is a dictionary describing labels associated with the signal. Typically contains a `'segments'`key, which maps to a list. Each element in the list is a dictionary defining a labeled segment:
      - `'start': float`: Start time of the segment (in seconds).
      - `'end': float`: End time of the segment (in seconds).
      - `'value': Dict`: A dictionary containing all types of true labels for this time segment. The keys of this dictionary are label type names (e.g., 'sleep_stage', 'event_type'), and the values are dictionaries containing the actual label data, like `{'data': label_value}`.
    - Top-level key `'meta'` (required): Its value is a dictionary containing any other metadata related to this record (e.g., original filename, subject ID).
  - **Example Return Structure**:

```python
{
    'signals': {
        'eeg': {'data': eeg_data_array, 'channels': ['C3', 'C4'], 'freq': 250.0},
        'ecg': {'data': ecg_data_array, 'channels': ['ECG'], 'freq': 500.0}
    },
    'labels': {
        'segments': [ # If labels are event/segment-based
            {'start': 0.5, 'end': 2.3, 'value': {'sleep_stage': {'data': 1}, 'arousal': {'data': 0}}},
            {'start': 5.0, 'end': 10.0, 'value': {'sleep_stage': {'data': 2}}}
        ]
    'meta': {
        'file_name': str(record), # Assuming record is a filename
        # ... other metadata ...
    }
}
```

- `process_record(self, signals: Dict, labels: Dict, meta: Dict, **kwargs) -> Generator[Dict, None, None]`

  - **Function**: This method is a **generator**. It receives the `signals`, `labels`, and `meta` dictionaries unpacked from `read_record`'s output (plus other `kwargs` from `__init__`), and is responsible for the core processing of this data.
  - **Typical Workflow**:
    1. Optionally apply `before_segment_transform` to `signals`.
    2. Call `self.segment_split(processed_signals, labels)` to segment the data by time.
    3. Iterate through each segment:
       - Get the segment's signals (`seg_signals`), labels (`seg_labels`), and time information (`seg_info`).
       - Generate a unique `segment_id` for this segment (e.g., using `self.get_segment_id(meta['file_name'], segment_index)`).
       - Optionally apply `offline_signal_transform` to `seg_signals`.
       - Optionally apply `offline_label_transform` to `seg_labels`.
       - **Populate `seg_info`**: Using `meta` information and helper methods like `get_subject_id`, add metadata such as `subject_id`, `session_id`, `trial_id`, `segment_id`, etc., to `seg_info`.
       - Call `self.assemble_segment(key=segment_id, signals=seg_signals, labels=seg_labels, info=seg_info)` to build the final output dictionary.
       - `yield` the dictionary returned by `assemble_segment`.
  - **Parameters**:
    - `signals` (`Dict`): The `'signals'` part from the dictionary returned by `read_record`.
    - `labels` (`Dict`): The `'labels'` part from the dictionary returned by `read_record`.
    - `meta` (`Dict`): The `'meta'` part from the dictionary returned by `read_record`.
    - `**kwargs`: Other parameters passed from `BaseDataset`'s `__init__` and `handle_record`.
  - **`yield` Dictionary Structure (Not user-defined)**: Each yielded dictionary represents a data unit (typically a segment) to be stored. Its structure is determined by the `assemble_segment` method and must conform to the expectations of `BaseDataset`'s internal writing methods (`write_signal`, `write_label`, `write_info`). Broadly, it looks like:
    - `'key': str`: A unique segment ID for storage.
    - `'signals': Dict[str, Dict]` (optional): Dictionary of processed signals. Each inner signal dictionary contains `'data'`, `'channels'`, `'freq'`, and an `'info'` dictionary (with `'sample_ids'`, `'windows'`).
    - `'labels': Dict[str, Dict]` (optional): Dictionary of processed labels. Each inner label dictionary contains `'data'` and an `'info'` dictionary (with `'sample_ids'`, optional `'windows'`).
    - `'info': Dict` (required): Metadata for this data segment, for `MetaInfoIO`. **Must include `'sample_ids': List[str]`**, and other identifiers like `subject_id`, `session_id`, etc..
  - **Example Logic Framework**:

  ```python
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
              segment_id = self.get_segment_id(meta['file_name'], idx)
              seg_signals = self.apply_transform(self.offline_signal_transform, seg_signals)
              seg_label = self.apply_transform(self.offline_label_transform, seg_label)
              if seg_signals is None or seg_label is None:
                  print(f"Skip segment {segment_id} due to transform error.")
                  continue
              
              seg_info.update({
                  'subject_id': self.get_subject_id(meta['file_name']),
                  'session_id': self.get_session_id(meta['file_name']),
                  'segment_id': self.get_segment_id(meta['file_name'], idx),
                  'trial_id': self.get_trial_id(idx),
              })
              yield self.assemble_segment(
                  key=segment_id,
                  signals=seg_signals,
                  labels=seg_label,
                  info=seg_info,
              )
  ```

### 2. Optionally Overridable Methods

The following methods provide default implementations, but you can override them according to specific dataset characteristics and processing needs:

- `segment_split(self, signals: Dict, labels: Dict) -> List[Dict]`:
  - The default implementation segments all signals in `signals` based on the `start` and `end` times (in seconds) found in the `labels['segments']` list. `start_offset` and `end_offset` are applied.
  - If your data segmentation logic is different (e.g., fixed-length sliding windows, or labels are not given in time segments), you need to override this method.
  - It should return a list, where each element is a dictionary representing a segment (containing segmented `'signals'`, `'labels'`, and time period `'info'`).
- `get_subject_id(**kwargs) -> str`, `get_session_id(**kwargs) -> str`, `get_trial_id(**kwargs) -> str`,`get_segment_id(**kwargs)->str`:
  - These methods (static or instance, depending on your design) are used to extract subject ID, session ID, trial ID, etc., from parameters passed to `process_record` (e.g., from the `meta` dictionary, the `record` identifier, or as in your example, from filename and index).
  - The default implementations in `BaseDataset` return `"0"`. If your data contains these hierarchical levels, you **must** override them to correctly extract these IDs for use in building the `info` dictionary within `process_record`.

### 3. Methods Not Usually Overridden (Internal Helper Methods)

- `assemble_segment(...) -> Dict`:
  - Responsible for assembling the processed single segment's `signals`, `labels`, `key`, and `info` into the standard dictionary structure that `process_record` ultimately `yield`s. It particularly handles the internal `'info'` fields within `signals` and `labels` (like `sample_ids`, `windows`). **This method has a fixed output structure to match data writing mechanisms and should generally not be overridden by users**.
- `assemble_sample(self, signals: Dict, labels: Dict) -> Dict`:
  - Called within `__getitem__`, it assembles the `signals` and `labels` read from disk into the single sample format expected by a DataLoader. The default implementation places data from all signal types and label types directly at the top level of the returned dictionary. **This method also has specific expected behavior for DataLoader compatibility and should generally not be overridden by users**.

### 4. Data Loading and Online Transforms

- `__getitem__(self, index: int) -> Dict`:
  - `BaseDataset` implements this method. It retrieves metadata for a sample from `self.info` (usually a `pd.DataFrame`) based on the index, then loads the corresponding signal and label data from disk (or cache) using `read_signal` and `read_label`.
  - After loading, `online_signal_transform` and `online_label_transform` are applied respectively.
  - Finally, `assemble_sample` is called to assemble and return the sample.
- `online_signal_transform` / `online_label_transform`:
  - These transforms, set in `__init__`, are dynamically applied each time data is requested via `__getitem__`. They are suitable for operations that should not be cached or whose results might vary per call.

[`Back to Top`](#tyeedataset)

## BCICIV2ADataset

```python
class BCICIV2ADataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV_2a', # 指向包含所有被试数据的BCICIV_2a数据集根目录
        start_offset: float = 2.0,
        end_offset: float = 6.0,
        include_end: bool = False,
        before_segment_transform: Union[None, Callable] = None,
        offline_signal_transform: Union[None, Callable] = None,
        offline_label_transform: Union[None, Callable] = None,
        online_signal_transform: Union[None, Callable] = None,
        online_label_transform: Union[None, Callable] = None,
        io_path: Union[None, str] = None, # 预处理后数据的缓存路径
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        lazy_threshold: int = 128,
        verbose: bool = True,
    ) -> None:
```

The `BCICIV2ADataset` class is used for handling the BCI Competition IV Dataset 2a. It inherits from `BaseDataset` and is responsible for reading raw Electroencephalography (EEG) and Electrooculography (EOG) data in `.gdf` format, as well as the corresponding label files in `.mat` format. This dataset typically includes multiple subjects, each with training ('T') and evaluation ('E') sessions. These sessions contain trials corresponding to four motor imagery tasks (left hand, right hand, foot, tongue). Data segments are usually extracted based on event markers (event ID '768') in the GDF files, combined with specified offsets.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the BCI Competition IV Dataset 2a. This directory should contain the raw data files for all subjects (e.g., `A01T.gdf`, `A01T.mat`, `A02T.gdf`, `A02T.mat`, etc.). Defaults to `'./BCICIV_2a'`.
- **start_offset** (`float`): The start time (in seconds) relative to the event marker, used for segment extraction. Defaults to `2.0` seconds.
- **end_offset** (`float`): The end time (in seconds) relative to the event marker, used for segment extraction. Defaults to `6.0` seconds.
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data. If specified, preprocessed data will be stored here and can be loaded directly during subsequent instantiations.
- **offline_signal_transform**, **offline_label_transform** (`Union[None, Callable]`): Transformations applied during the data preprocessing stage (offline), the results of which are cached.
- **online_signal_transform**, **online_label_transform** (`Union[None, Callable]`): Transformations applied during the data loading stage (online); results are not cached.
- Other parameters such as `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in [`BaseDataset`]().

**Dataset Characteristics**

- Supported signal types:
  - `eeg`: Electroencephalography signals.
  - `eog`: Electrooculography signals.
- Sampling rate:
  - The sampling rate for all signals (EEG and EOG) is **250 Hz**.
- Channel information:
  - `eeg`: Contains 22 standard EEG channels.
  - `eog`: Contains 3 EOG channels, typically named `['left', 'central', 'right']` after processing.
- Label information:
  - Labels are loaded from `.mat` files and correspond to the motor imagery tasks.
  - Original labels are 1, 2, 3, 4, which are converted to **0, 1, 2, 3** in this dataset.
  - Class correspondence:
    - `0`: Left hand
    - `1`: Right hand
    - `2`: Foot
    - `3`: Tongue

**Usage Example**

~~~python
from dataset.bciciv2a_dataset import BCICIV2ADataset
from dataset.transform import Cheby2Filter, Select
offline_signal_transform = [
     Cheby2Filter(l_freq=4, h_freq=40, source='eeg', target='eeg'),
     Select(key=['eeg']),
]
dataset = BCICIV2ADataset(
         root_path='./BCICIV_2a',
         io_path='./tyee_bciciv2a',
         io_mode='hdf5',
         io_chunks=750,
         offline_signal_transform=offline_signal_transform
)
  print(dataset[0]
~~~

[`Back to Top`](#tyeedataset)

## BCICIV4Dataset

```python
class BCICIV4Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './BCICIV4',
        start_offset: float = 0.0,
        end_offset: float = 0.0,
        include_end: bool = False,
        before_segment_transform: Union[None, List[Callable]] = None,
        offline_signal_transform: Union[None, List[Callable]] = None,
        offline_label_transform: Union[None, List[Callable]] = None,
        online_signal_transform: Union[None, List[Callable]] = None,
        online_label_transform: Union[None, List[Callable]] = None,
        io_path: Union[None, str] = None,
        io_size: int = 1048576,
        io_chunks: int = None,
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        verbose: bool = True,
    ) -> None:
```

The `BCICIV4Dataset` class is used to process Electrocorticography (ECoG) data from BCI Competition IV Dataset 4. This dataset contains multiple subjects, and the data file for each subject (in `.mat` format, e.g., `sub1_comp.mat`) includes ECoG signals from training and testing phases, as well as corresponding finger flexion data (Dataglove, used as labels here). This class inherits from `BaseDataset` and implements data reading and preprocessing logic specific to this dataset.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for BCI Competition IV Dataset 4. This directory should contain the `*_comp.mat` and `*_testlabels.mat` files for each subject. Defaults to `'./BCICIV4'`.
- **start_offset** (`float`): Offset (in seconds) relative to the start time of each segment (here referring to training and testing segments). Defaults to `0.0` seconds.
- **end_offset** (`float`): Offset (in seconds) relative to the end time of each segment. Defaults to `0.0` seconds.
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data. If specified, preprocessed data will be stored here and can be loaded directly during subsequent instantiations.
- **offline_signal_transform**, **offline_label_transform** (`Union[None, List[Callable]]`): A list of transformations applied during the data preprocessing stage (offline), the results of which are cached.
- **online_signal_transform**, **online_label_transform** (`Union[None, List[Callable]]`): A list of transformations applied during the data loading stage (online); results are not cached.
- Other parameters such as `io_size`, `io_chunks`, `io_mode`, `num_worker`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Supported signal types:
  - `ecog`: Electrocorticography signals. Data is loaded from the `train_data` and `test_data` fields in the `.mat` file and concatenated.
- Sampling rate:
  - `ecog`: **1000 Hz**.
  - `dg` (Dataglove, processed as labels): Original sampling rate is **1000 Hz**; can be resampled in offline transformations.
- Channel information:
  - `ecog`: Channels are named with numerical indices starting from "0" in the dataset.
- Label Information (Dataglove):
  - Label data consists of continuous multi-channel Dataglove finger flexion signals.
  - In `read_record`, the raw ECoG data is logically divided into a "training segment" and a "testing segment," corresponding to `train_data` and `test_data` from the `.mat` file, respectively.
  - The Dataglove data (`train_dg` or `test_dg`) for each logical segment (training/testing segment) serves as its label value, with the structure `{'dg': {'data': dataglove_array, 'freq': 1000}}`.
  - Consequently, the `segment_split` method of `BaseDataset` (if the default is used) will segment the ECoG and corresponding Dataglove data based on the time ranges of these two logical segments. `start_offset` and `end_offset` are applied to these two main segments.

**Usage Example**

~~~python
from dataset.bciciv4_dataset import BCICIV4Dataset
from dataset.transform import Compose, NotchFilter, Filter, ZScoreNormalize, RobustNormalize, Reshape,\
                              CWTSpectrum, Downsample, Crop, Interpolate, MinMaxNormalize, CommonAverageRef,\
                              Transpose
from dataset.transform.slide_window import SlideWindow

offline_signal_transform = [
    
    ZScoreNormalize(epsilon=0, axis=1, source='ecog', target='ecog'),
    CommonAverageRef(axis=0, source='ecog', target='ecog'),
    Filter(l_freq=40, h_freq=300, source='ecog', target='ecog'),
    NotchFilter(freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450], source='ecog', target='ecog'),
    CWTSpectrum(freqs=np.logspace(np.log10(40), np.log10(300), 40), output_type='power', n_jobs=6, source='ecog', target='ecog'),
    Downsample(desired_freq=100, source='ecog', target='ecog'),
    Crop(crop_right=20, source='ecog', target='ecog'),
    SlideWindow(window_size=256, stride=1, source='ecog', target='ecog'),
]
offline_label_transform = [
        Downsample(desired_freq=25, source='dg', target='dg'),
        Interpolate(desired_freq=100, kind='cubic', source='dg', target='dg'),
        Crop(crop_left=20, source='dg', target='dg'),
        SlideWindow(window_size=256, stride=1, source='dg', target='dg'),
]
dataset = BCICIV4Dataset(root_path='/home/lingyus/data/BCICIV4/sub1',
                        io_path='/home/lingyus/data/BCICIV4/sub1/processed_test',
                        offline_signal_transform=offline_signal_transform,
                        offline_label_transform=offline_label_transform,
                        io_mode='hdf5',
                        io_chunks=256,
                        num_worker=8)
~~~

[`Back to Top`](#tyeedataset)

## CinC2018Dataset

```python
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
```

The `CinC2018Dataset` class is used to process the PhysioNet/CinC Challenge 2018 dataset, which focuses on automatic sleep stage classification using multi-channel physiological signals. This class inherits from `BaseDataset` and implements logic for loading and preprocessing data from specific file formats: `.hea` for header files, `.mat` for signal data, and `*-arousal.mat` for arousal/sleep stage labels.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the PhysioNet Challenge 2018 training dataset. This directory should contain subfolders for individual recordings, with each subfolder containing the corresponding `.hea`, `.mat` (signals), and `*-arousal.mat` (labels) files. Defaults to `'./challenge-2018/training'`.
- **start_offset** (`float`): Offset (in seconds) relative to the start time of each 30-second sleep epoch marker. In this dataset, since the labels themselves define the epochs, this parameter would typically be used with `BaseDataset`'s default `segment_split`, but `CinC2018Dataset` overrides `segment_split`. Defaults to `0.0` seconds.
- **end_offset** (`float`): Offset (in seconds) relative to the end time of each 30-second sleep epoch marker. Defaults to `0.0` seconds.
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - `.hea` file: Contains header information for the recording, such as signal names, sampling frequency, etc.
  - `.mat` file (e.g., `tr03-0005.mat`): Contains the actual multi-channel physiological signal data.
  - `*-arousal.mat` file: Contains annotations for sleep stages and arousal events.
- Supported signal types (loaded from `.mat` files and grouped according to `.hea` file information):
  - `eeg`: Electroencephalography signals (typically the first 6 channels, e.g., 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1').
  - `eog`: Electrooculography signals (typically the 7th channel, e.g., 'E1-M2').
  - `emg`: Electromyography signals (typically the 8th channel, e.g., 'Chin1-Chin2').
  - `abd`: Abdominal respiration signal.
  - `chest`: Thoracic respiration signal.
  - `airflow`: Airflow signal.
  - `sao2`: Blood oxygen saturation signal.
  - `ecg`: Electrocardiogram signal.
- Sampling rate:
  - Dynamically read from each recording's `.hea` file (via the `import_signal_names` method). For this dataset, it is typically **200 Hz**.
- Channel information:
  - Read from the `.hea` file. Specific channel names and counts may vary by recording but generally follow the signal type groupings mentioned above.
- Label information:
  - Sleep stage labels are extracted from `*-arousal.mat` files.
  - Labels are processed into 30-second sleep epochs.
  - Sleep stage mapping:
    - "wake": 0
    - "nonrem1": 1
    - "nonrem2": 2
    - "nonrem3": 3
    - "rem": 4
- **Segmentation (specific behavior of `segment_split`)**: This class overrides the `segment_split` method. It aggregates all valid 30-second sleep epoch data within a single recording file and then stacks them separately by signal type and label type. Therefore, unless further transformations like `SlideWindow` are applied, `BaseDataset`'s `process_record` method, when processing each raw recording file, primarily operates on this aggregated "whole segment" data (where the first dimension represents the number of epochs).

**Usage Example**

~~~python
from dataset.cinc2018_dataset import CinC2018Dataset
from dataset.transform import SlideWindow, Resample,Compose,Select,Concat,Squeeze,PickChannels
before_segment_transform = [
   PickChannels(channels=["C3-M2", "C4-M1", "O1-M2", "O2-M1"], source='eeg', target='eeg'),
    Concat(axis=0, source=['eeg', 'eog'], target='ss'),
    Concat(axis=0, source=['chest','sao2','abd'], target='resp'),
    Resample(desired_freq=256,source='ss', target='ss'),
    Resample(desired_freq=256,source='resp', target='resp'),
    Resample(desired_freq=256,source='ecg', target='ecg'),
    Select(key=['ss','resp','ecg']),
]
offline_signal_transform = [
    SlideWindow(window_size=1, stride=1, axis=0, source='ss', target='ss'),
    SlideWindow(window_size=1, stride=1, axis=0, source='resp', target='resp'),
    SlideWindow(window_size=1, stride=1, axis=0, source='ecg', target='ecg'),
]
offline_label_transform = [
    SlideWindow(window_size=1, stride=1, axis=0, source='stage', target='stage'),
]
online_signal_transform = [
    Squeeze(axis=0, source='ss', target='ss'),
    Squeeze(axis=0, source='resp', target='resp'),
    Squeeze(axis=0, source='ecg', target='ecg'),
]
online_label_transform = [
    Squeeze(axis=0, source='stage', target='stage'),
]
datasets = CinC2018Dataset(
    root_path='/mnt/ssd/lingyus/challenge-2018-split/train',
    io_path='/mnt/ssd/lingyus/tyee_cinc2018/train',
    io_mode='hdf5',
    io_chunks=1,
    before_segment_transform=before_segment_transform,
    offline_signal_transform=offline_signal_transform,
    offline_label_transform=offline_label_transform,
    online_signal_transform=online_signal_transform,
    online_label_transform=online_label_transform,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## DaLiADataset

```python
class DaLiADataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './PPG_FieldStudy',
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
```

The `DaLiADataset` class is used for loading and processing physiological signals from the PPG_DaLiA dataset, such as Photoplethysmography (PPG) and Accelerometer (ACC) data, along with corresponding Heart Rate (HR) labels obtained from `.pkl` files. This class inherits from `BaseDataset`.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the PPG_DaLiA dataset. This directory should contain the data files for individual subjects (typically `S*.pkl`). Defaults to `'./PPG_FieldStudy'`.
- **start_offset** (`float`): An offset (in seconds) applied to the start time of segments. Given that DaLiA data initially treats an entire recording as a single segment, this offset (if non-zero) would apply to the beginning of the whole recording. Defaults to `0`.
- **end_offset** (`float`): An offset (in seconds) applied to the end time of segments. Defaults to `0`.
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Supported signal types:
  - `ppg`: Photoplethysmography signals, typically from a wrist-worn device.
  - `acc`: Accelerometer signals, typically from a wrist-worn device (usually 3-axis).
- Sampling rate:
  - `ppg`: **64 Hz**.
  - `acc`: **32 Hz**.
- Label information:
  - `hr`: Heart rate data, where each HR data point corresponds to an 8-second window with a stride of 2 seconds.

**Usage Example**

~~~python
from dataset.dalia_dataset import DaLiADataset
from dataset.transform import WindowExtract,SlideWindow, ForEach, Filter,Detrend,\
                            ZScoreNormalize, Lambda, Resample, Pad, FFTSpectrum,\
                            Stack, ExpandDims, Crop, Select, Compose, Mean
iir_params = dict(order=4, ftype='butter')
offline_signal_trasnform=[
    Compose(
        transforms=[
            Detrend(),
            Filter(l_freq=0.1, h_freq=18, method='iir', phase='forward', iir_params=iir_params),
            Mean(axis=0),
            ExpandDims(axis=-1),
            ZScoreNormalize(epsilon=1e-10),
            SlideWindow(window_size=1280, stride=128, axis=0),
        ],source='ppg', target='ppg_time'
    ),
    Select(key=['ppg_time']),
]
offline_label_trasnform=[
    Compose(
        transforms=[
            Crop(crop_left=6),
            SlideWindow(window_size=1, stride=1, axis=0),
        ], source='hr', target='hr'
    )
]

dataset = DaLiADataset(
    root_path='/mnt/ssd/lingyus/ppg_dalia/PPG_FieldStudy',
    io_path='/mnt/ssd/lingyus/tyee_ppgdalia/train',
    offline_signal_transform=offline_signal_trasnform,
    offline_label_transform=offline_label_trasnform,
    num_worker=4,
    io_chunks=320,
)
~~~

[`Back to Top`](#tyeedataset)

## DEAPDataset

```python
class DEAPDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './DEAP/data_preprocessed_python',
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
```

The `DEAPDataset` class is used to handle the DEAP (Database for Emotion Analysis using Physiological Signals) dataset. This dataset contains recordings of various physiological signals from 32 participants while they watched 40 music videos (each 1 minute long), as well as their emotional ratings (Valence, Arousal, Dominance, Liking) for each video. This class inherits from `BaseDataset` and is responsible for loading data from preprocessed Python pickled files (`.dat`).

**Main Initialization Parameters**

- **root_path** (`str`): Path to the `data_preprocessed_python` folder of the DEAP dataset. This folder should contain the `.dat` files for all subjects (e.g., `s01.dat`, `s02.dat`, ..., `s32.dat`). Defaults to `'./DEAP/data_preprocessed_python'`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`. Note: The preprocessed version of the DEAP dataset has already undergone downsampling, filtering, etc. `start_offset` and `end_offset` are typically kept at their default value of 0 for this dataset, as the data is already segmented into trials.

**Dataset Characteristics**

- Raw data files:

  - Each subject's data is stored in a `.dat` file, which is a Python pickled object containing all physiological signals and emotion ratings for the 40 videos (trials) watched by that subject.

- Supported signal types (loaded from 

  ```
  .dat
  ```

   files):

  - `eeg`: Electroencephalography signals (32 channels).
  - `eog`: Electrooculography signals (2 channels: hEOG, vEOG).
  - `emg`: Electromyography signals (2 channels: zEMG, tEMG - corresponding to Zygomaticus Major and Trapezius muscles, respectively).
  - `gsr`: Galvanic Skin Response.
  - `resp`: Respiration signal.
  - `ppg`: Photoplethysmograph.
  - `temp`: Skin temperature.

- Sampling rate:

  - The sampling rate for all signals is **128 Hz** (this is the rate provided in the preprocessed version of the DEAP dataset).

- Channel information:

  - `eeg`: 32 EEG channels, following the international 10-20 system arrangement (specifically in the order: FP1, AF3, F3, F7, FC5, FC1, C3, T7, CP5, CP1, P3, P7, PO3, O1, OZ, PZ, FP2, AF4, FZ, F4, F8, FC6, FC2, CZ, C4, T8, CP6, CP2, P4, P8, PO4, O2).
  - `eog`: 2 channels (`hEOG`, `vEOG`).
  - `emg`: 2 channels (`zEMG`, `tEMG`).
  - `gsr`, `resp`, `ppg`, `temp`: Each is a single-channel signal.

- Label information:

  - Each trial (video) corresponds to a set of emotion ratings, ranging from 1 to 9.
  - `valence`: Valence (pleasantness).
  - `arousal`: Arousal.
  - `dominance`: Dominance (control).
  - `liking`: Liking.

- **Segmentation (behavior of `segment_split`)**: The raw data in the DEAP dataset is already segmented by trial (each video, 60 seconds of effective data). The `read_record` method treats each trial as an independent recording unit. The `segment_split` method (overridden in this class) processes this pre-segmented trial data, treating the data of each trial as one segment and extracting the corresponding signals and labels. Therefore, unless an offline transformation like `SlideWindow` is applied to further subdivide this 60-second trial data, `process_record` will iterate over each complete 60-second trial data block.

**Usage Example**

~~~python
from dataset.deap_dataset import DEAPDataset
from dataset.transform import MinMaxNormalize, SlideWindow,Compose, Mapping, OneHotEncode, Concat,Select,Round, ToNumpyInt32
offline_signal_transform = [
    Concat(axis=0, source=['gsr', 'resp', 'ppg', 'temp'], target='mulit4'),
    Compose([
        MinMaxNormalize(axis=-1),
        SlideWindow(window_size=128*5, stride=128*3)],
        source='mulit4', target='mulit4'),
    Select(key=['mulit4']),
]
offline_label_transform = [
    Compose([
        Round(),
        ToNumpyInt32(),
        Mapping(mapping={
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,}),
        # OneHotEncode(num=9),
    ], source='arousal', target='arousal'),
    Select(key=['arousal']),
]
import numpy as np
import pandas as pd
dataset = DEAPDataset(
    root_path='/mnt/ssd/lingyus/test',
    io_path='/mnt/ssd/lingyus/tyee_deap/train',
    io_chunks=640,
    offline_label_transform=offline_label_transform,
    offline_signal_transform=offline_signal_transform,
    io_mode='hdf5',
    num_worker=4,
)
~~~

[`Back to Top`](#tyeedataset)

## KaggleERNDataset

```python
class KaggleERNDataset(BaseDataset):
    def __init__(
        self,
        root_path: str,
        start_offset: float = -0.7,
        end_offset: float = 1.3,
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
```

The `KaggleERNDataset` class is used to handle data from the Kaggle "Grasp-and-Lift EEG Detection" competition. This dataset focuses on detecting Error-Related Negativity (ERN). The class inherits from `BaseDataset` and implements logic for loading and preprocessing EEG signals and corresponding labels from specific CSV file formats.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the Kaggle ERN dataset. This directory should contain the session data CSV files for each subject (e.g., `Data_S02_Sess01.csv`) as well as the corresponding label files (`TrainLabels.csv` or `true_labels.csv`).
- **start_offset** (`float`): The start time (in seconds) relative to the feedback event (`FeedBackEvent`), used for segment extraction. Defaults to `-0.7` seconds (i.e., 0.7 seconds before the event).
- **end_offset** (`float`): The end time (in seconds) relative to the feedback event, used for segment extraction. Defaults to `1.3` seconds (i.e., 1.3 seconds after the event).
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Signal data is stored in CSV files named in the format `Data_SXX_SessYY.csv`.
  - Label data is loaded from `TrainLabels.csv` or `true_labels.csv` based on the subject ID (training set or test set) (these tables need to be placed in the corresponding folder).
- Supported signal types:
  - `eeg`: Electroencephalography signals.
- Sampling rate:
  - `eeg`: **200 Hz**.
- Channel information:
  - `eeg`: Contains 58 EEG channels. The specific list of channel names is hardcoded in the class (FP1, FP2, AF7, ..., O1, O2).
- Label information:
  - Binary labels (0 or 1), indicating whether the task feedback was correct (`0`) or incorrect (`1`).

**Usage Example**

~~~python
from dataset.kaggleern_dataset import KaggleERNDataset
from dataset.transform import MinMaxNormalize, Offset, Scale, PickChannels

offline_signal_transform = [
    MinMaxNormalize(source='eeg', target='eeg'),
    Offset(offset=-0.5, source='eeg', target='eeg'),
    Scale(scale_factor=2.0, source='eeg', target='eeg'),
    PickChannels(channels=['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2'], source='eeg', target='eeg')
]
dataset = KaggleERNDataset(
    root_path='/mnt/ssd/lingyus/KaggleERN/train',
    io_path='/mnt/ssd/lingyus/tyee_kaggleern/train',
    io_chunks= 400,
    io_mode='hdf5',
    offline_signal_transform=offline_signal_transform,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## MITBIHDataset

~~~python
class MITBIHDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './mit-bih-arrhythmia-database-1.0.0',
        start_offset: float = -64/360, # 约 -0.178s
        end_offset: float = 64/360,   # 约 +0.178s (总窗口约 128 个点 @ 360Hz)
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
~~~

The `MITBIHDataset` class is used to handle the MIT-BIH Arrhythmia Database. This dataset contains two-channel electrocardiogram (ECG) recordings with detailed arrhythmia type annotations. The class inherits from `BaseDataset` and implements logic for loading and preprocessing data from WFDB format files. Data segments are typically extracted around each detected R-peak, based on specified offsets.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the MIT-BIH Arrhythmia Database. This directory should contain WFDB format files (e.g., `100.dat`, `100.hea`, `100.atr`, etc.). Defaults to `'./mit-bih-arrhythmia-database-1.0.0'`.
- **start_offset** (`float`): The start time (in seconds) relative to each R-peak (as an event marker), used for segment extraction. Defaults to `-64/360` seconds (approximately -0.178 seconds), aimed at capturing the signal before the R-peak.
- **end_offset** (`float`): The end time (in seconds) relative to each R-peak, used for segment extraction. Defaults to `64/360` seconds (approximately +0.178 seconds). Combined with the default `start_offset`, this typically forms a window of 128 samples (at a 360Hz sampling rate) centered around the R-peak.
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Data is in WFDB format, with each recording typically including `.dat` (signal), `.hea` (header), and `.atr` (annotation) files. `root_path` should point to the top-level directory containing these recording files.
- Supported signal types:
  - `ecg`: Electrocardiogram signals.
- Sampling rate:
  - Dynamically read from each recording's `.hea` file. For the MIT-BIH dataset, this is typically **360 Hz**.
- Channel information:
  - Typically 2-channel ECG signals. Common channels include 'MLII' (a modified Lead II) and one of 'V1', 'V2', 'V4', or 'V5'. Channel names are read from the `.hea` file.
- Label information:
  - Arrhythmia type annotations (symbols) are read from the `.atr` file. Each label (symbol) is associated with the position of an R-peak, and the label is the annotation symbol corresponding to that R-peak.

**Usage Example**

~~~python
from dataset.mit_bih_dataset import MITBIHDataset
from dataset.transform import PickChannels, Mapping, ZScoreNormalize

before_segment_transform = [
    PickChannels(channels=['MLII'], source='ecg', target='ecg'),
    ZScoreNormalize(axis=-1, source='ecg', target='ecg'),
]
offline_label_transform = [
    Mapping(mapping={
        'N': 0,
        'V': 1,
        '/': 2,
        'R': 3,
        'L': 4,
        'A': 5,
        '!': 6,
        'E': 7,
    }, source='label', target='label'),
]

dataset = MITBIHDataset(
    root_path='/mnt/ssd/lingyus/test',
    io_path='/mnt/ssd/lingyus/tyee_mit_bih/train',
    before_segment_transform=before_segment_transform,
    offline_label_transform=offline_label_transform,
    io_mode='hdf5',
    io_chunks=128,
    num_worker=4,
)
print(len(dataset))
print(dataset[0]['ecg'].shape)
~~~

[`Back to Top`](#tyeedataset)

## NinaproDB5Dataset

~~~python
class NinaproDB5Dataset(BaseDataset):
    def __init__(
        self,
        # NinaproDB5 specific parameters
        wLen: float = 0.25,
        stepLen: float = 0.05,
        balance: bool = True,
        include_transitions: bool = False,
        # BaseDataset parameters
        root_path: str = './NinaProDB5',
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
~~~

The `NinaproDB5Dataset` class is used to handle the Ninapro DB5 dataset. This dataset contains surface Electromyography (sEMG) signals from 10 intact subjects performing approximately 50 different hand and wrist movements. This class inherits from `BaseDataset` and implements logic for loading data from `.mat` files, and generating data segments and corresponding gesture labels based on sliding window parameters.

**Main Initialization Parameters**

- **wLen** (`float`): Length of the sliding window (in seconds). Defaults to `0.25` seconds.
- **stepLen** (`float`): Step size (stride) of the sliding window (in seconds). Defaults to `0.05` seconds.
- **balance** (`bool`): Whether to perform class balancing on the generated data windows (primarily for the rest state '0' gesture). If `True`, it attempts to reduce the number of rest state windows to be roughly comparable to the number of windows for other gestures. Defaults to `True`.
- **include_transitions** (`bool`): When processing labels, whether to include transition windows between gestures. If `True`, windows containing two gestures will be specially handled; if `False`, the gesture at the start of the window is usually taken as the label for that window. Defaults to `False`.
- **root_path** (`str`): The root directory path for the Ninapro DB5 dataset. This directory should recursively contain the `.mat` files for each subject (e.g., `S1_A1_E1.mat`, `S1_A1_E2.mat`, etc.). Defaults to `'./NinaProDB5'`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other `BaseDataset` parameters such as `start_offset`, `end_offset`, `include_end` (these are typically kept at their default value of 0 in this dataset, as segmentation is mainly controlled by `wLen` and `stepLen`), as well as various transform callback functions and IO parameters, have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Data for each exercise of each subject is stored in a `.mat` file. The `set_records` method recursively finds all `.mat` files under `root_path`.
- Supported signal types:
  - `emg`: Surface Electromyography signals.
- Sampling rate:
  - Dynamically read from the 'frequency' field in each `.mat` file. For Ninapro DB5, this is **200 Hz**.
- Channel information:
  - The number of EMG channels is obtained from the data (`emg_data.shape[0]`). Ninapro DB5 typically contains 16 EMG channels. These channels are named with numerical indices starting from "1" in this class.
- Label information:
  - `'gesture'`: Gesture labels, obtained from the 'restimulus' field in the `.mat` file.
  - Original labels represent different gestures or the rest state. The `read_record` method adjusts the original label values based on the movement category in the filename (Exercise A, B, C correspond to different gesture sets) to generate a unique gesture ID across the entire dataset. The rest state is usually labeled as 0.

**Usage Example**

~~~python
from dataset.ninapro_db5_dataset import NinaproDB5Dataset
from dataset.transform import Mapping, Filter
onffline_label_transform = [
    Mapping(mapping={
        0: 0,
        17: 1,
        18: 2,
        20: 3,
        21: 4,
        22: 5,
        25: 6,
        26: 7,
        27: 8,
        28: 9,
    }, source='gesture', target='gesture'),
]
offline_signal_transform = [
    Filter(h_freq=None, l_freq=5.0, method='iir', iir_params=dict(order=3, ftype='butter', padlen=12), phase='zero', source='emg', target='emg'),
]
dataset = NinaproDB5Dataset(
    root_path='/mnt/ssd/lingyus/NinaproDB5E2',
    io_path='/mnt/ssd/lingyus/tyee_ninapro_db5/train',
    offline_label_transform=onffline_label_transform,
    offline_signal_transform=offline_signal_transform,
    io_mode='hdf5',
    num_worker=4,
)

~~~

[`Back to Top`](#tyeedataset)

## PhysioP300Dataset

~~~python
class PhysioP300Dataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './lingyus/erp-based-brain-computer-interface-recordings-1.0.0',
        start_offset: float = -0.7,
        end_offset: float = 1.3,
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
~~~

The `PhysioP300Dataset` class is used to handle the "ERP-based Brain-Computer Interface Recordings" dataset from PhysioNet, which is commonly used in Brain-Computer Interface (BCI) research, such as P300 spellers. This class inherits from `BaseDataset` and implements logic for loading EEG signals and event annotations from `.edf` files, and for extracting stimulus-related EEG segments and corresponding labels (target/non-target stimuli).

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the PhysioNet P300 dataset. This directory should recursively contain the `.edf` files for each subject and session. Defaults to `'./lingyus/erp-based-brain-computer-interface-recordings-1.0.0'` (please modify according to the actual path).
- **start_offset** (`float`): The start time (in seconds) relative to each stimulus event marker, used for segment extraction. Defaults to `-0.7` seconds (i.e., 0.7 seconds before the event).
- **end_offset** (`float`): The end time (in seconds) relative to each stimulus event marker, used for segment extraction. Defaults to `1.3` seconds (i.e., 1.3 seconds after the event).
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Signal data and event annotations are stored in `.edf` files. The `set_records` method recursively finds all `.edf` files under `root_path`.
- Supported signal types:
  - `eeg`: Electroencephalography signals.
- Sampling rate:
  - Dynamically read from the header information of each `.edf` file (`raw.info['sfreq']`). The original sampling rate for this dataset is typically **2048 Hz**.
- Channel information:
  - Read from the `.edf` file. Typically contains 64 EEG channels (e.g., Fp1, AF7, ..., O2, etc.; channel names are converted to uppercase).
- Label information:
  - Binary labels (0 or 1), indicating whether the presented stimulus is a target stimulus in the P300 task.
    - `1`: Target stimulus
    - `0`: Non-target stimulus

**Usage Example**

~~~python
from dataset.physiop300_dataset import PhysioP300Dataset
from dataset.transform import PickChannels, Resample, Filter, Scale, Baseline
channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
channels = [ch.upper() for ch in channels]
offline_signal_transform = [
    PickChannels(channels=channels, source='eeg', target='eeg'),
    Baseline(baseline_range=(0, 1435), axis=1, source='eeg', target='eeg'),
    Filter(l_freq=0, h_freq=120, method= 'iir', source='eeg', target='eeg'),
    Resample(desired_freq=256, pad="edge", source='eeg', target='eeg'),
    Scale(scale_factor=1e-3, source='eeg', target='eeg')
]
online_signal_transform = [
    Baseline(baseline_range=(0, 1434), axis=1, source='eeg', target='eeg'),
]
dataset = PhysioP300Dataset(
    root_path='/mnt/ssd/lingyus/erp-based-brain-computer-interface-recordings-1.0.0',
    io_path='/mnt/ssd/lingyus/tyee_physio_p300',
    io_chunks= 512,
    io_mode='hdf5',
    include_end=True,
    offline_signal_transform=offline_signal_transform,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## SEEDVFeatureDataset

~~~python
class SEEDVFeatureDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './SEED-V',
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
~~~

The `SEEDVFeatureDataset` class is used to handle pre-extracted feature data from the SEED-V dataset. SEED-V is a dataset for emotion recognition based on multimodal physiological signals (EEG and eye-tracking data). This class inherits from `BaseDataset` and implements logic for loading Differential Entropy (DE) features of EEG and eye-tracking features from `.npz` files, along with corresponding emotion labels.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the SEED-V feature dataset. This directory should contain two subdirectories: `EEG_DE_features` and `Eye_movement_features`, which store the `.npz` files for the respective features. Defaults to `'./SEED-V'`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data (although this class primarily loads already extracted features, the `BaseDataset` framework still uses this parameter).
- Other `BaseDataset` parameters such as `start_offset`, `end_offset`, `include_end` (these are typically kept at their default value of 0 in this dataset, as the data consists of feature sequences and the segmentation logic is specifically implemented in this class), as well as various transform callback functions and IO parameters, have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - EEG DE features and eye-tracking features are stored in `.npz` files under the `EEG_DE_features` and `Eye_movement_features` directories, respectively. Each `.npz` file typically corresponds to one experimental session for a subject and may contain features for multiple trials.
  - The `set_records` method matches pairs of `.npz` files from these two directories that have the same filename (excluding suffixes like experiment dates, etc.) (e.g., belonging to the same subject).
- Supported feature types:
  - `eeg`: Differential Entropy (DE) features of EEG signals.
  - `eog`: Eye-tracking features.
  - Note: These are feature vectors, not raw time-domain signals.
- Sampling rate:
  - For feature data, "sampling rate" in the traditional sense is not directly applicable. Features are typically extracted from specific time windows of the original signal. This class does not explicitly set a `'freq'` field for these feature data in `read_record`.
- Label information:
  - `'emotion'`: Emotion labels, loaded from the `label` field of the EEG feature files.
  - Emotion labels in the SEED-V dataset are typically integers representing different emotion categories (e.g., 0: neutral, 1: sad, 2: fear, 3: happy).
- The `segment_split` method is **overridden**: it uses the `start` and `end` indices (based on feature vector indices), which are pre-calculated for `labels['segments']` in `read_record`, to segment the concatenated overall feature data. This means that each "segment" ultimately processed by `process_record` corresponds to the feature set of one trial/video clip from the original data.

**Usage Example**

~~~python
from dataset.seedv_dataset import SEEDVFeatureDataset
from dataset.transform import SlideWindow

offline_signal_transform = [
    Log(epsilon=1, source='eog', target='eog'),
    SlideWindow(window_size=1, stride=1, axis=0, source='eeg', target='eeg'),
    SlideWindow(window_size=1, stride=1, axis=0, source='eog', target='eog')
]

offline_label_transform = [
    SlideWindow(window_size=1, stride=1, axis=0, source='emotion', target='emotion')
]
dataset = SEEDVFeatureDataset(
    root_path='/your/data/path',
    io_path='/your/io/path',
    io_chunks=1,
    io_mode='hdf5',
    offline_signal_transform=offline_signal_transform,
    offline_label_transform=offline_label_transform,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## SleepEDFCassetteDataset

~~~python
class SleepEDFCassetteDataset(BaseDataset):
    def __init__(
        self,
        # SleepEDFCassetteDataset specific parameters
        crop_wake_mins: int = 30,
        # BaseDataset parameters
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
~~~

The `SleepEDFCassetteDataset` class is used to handle the "sleep-cassette" part of the Sleep-EDF Database Expanded. This dataset contains polysomnography (PSG) recordings from healthy subjects, intended for sleep stage analysis. This class inherits from `BaseDataset` and implements logic for loading signals and sleep stage annotations from `.edf` files.

**Main Initialization Parameters**

- **crop_wake_mins** (`int`): If greater than 0, when processing raw recordings, data will be truncated starting `crop_wake_mins` minutes before the first sleep stage and `crop_wake_mins` minutes after the last sleep stage, to focus on the main sleep period. Defaults to `30` minutes.
- **root_path** (`str`): The root directory path for the Sleep-EDF "sleep-cassette" dataset. This directory should contain pairs of `SC*.edf` (PSG signals) and `SC*Hypnogram.edf` (sleep stage annotations) files. Defaults to `'./sleep-edf-database-expanded-1.0.0/sleep-cassette'`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other `BaseDataset` parameters such as `start_offset`, `end_offset`, `include_end` (these are typically kept at their default value of 0 in this dataset, as segmentation is mainly driven by MNE's 30-second epoch events), as well as various transform callback functions and IO parameters, have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Each recording includes one `*PSG.edf` file (containing multi-channel physiological signals) and a corresponding `*Hypnogram.edf` file (containing sleep stage annotations). The `set_records` method (using the `list_records` helper function) is responsible for finding and pairing these files.
- Supported signal types (loaded from `*PSG.edf` files):
  - `eeg`: Electroencephalography signals (2 channels: 'Fpz-Cz', 'Pz-Oz').
  - `eog`: Electrooculography signal (1 channel: 'horizontal').
  - `rsp`: Respiration signal (1 channel: 'oro-nasal' - oro-nasal airflow).
  - `emg`: Electromyography signal (1 channel: 'submental' - submental muscle).
  - `temp`: Body temperature signal (1 channel: 'rectal' - rectal temperature).
- Sampling rate:
  - Dynamically read from the header information of each recording's `.edf` file (`raw.info['sfreq']`). For "sleep-cassette" data, EEG, EOG, EMG are typically **100 Hz**, while respiration and body temperature signals are typically **1 Hz**.
- Label information:
  - `'stage'`: Sleep stage labels, extracted from annotations in the `*Hypnogram.edf` file.
  - Sleep stages are typically scored in 30-second epochs.
  - Label mapping (0-indexed):
    - `0`: 'W' (Wake)
    - `1`: '1' (NREM Stage 1, N1)
    - `2`: '2' (NREM Stage 2, N2)
    - `3`: '3' (NREM Stage 3, N3)
    - `4`: '4' (NREM Stage 4, N4) (In some scoring standards, N3 and N4 are combined into N3)
    - `5`: 'R' (REM, Rapid Eye Movement)
    - Undefined or movement artifact periods may be mapped to `-1` or ignored.
- **Segmentation (specific behavior of `segment_split`)**: This class overrides the `segment_split` method. It first extracts valid data segments for each signal type based on `labels['segments']` (generated by `read_record`, where each element represents a 30-second sleep epoch). Then, it stacks **all valid 30-second data segments** separately by signal type. Therefore, unless further transformations like `SlideWindow` are applied, `BaseDataset`'s `process_record` method, when processing each raw recording file, primarily operates on this aggregated "whole segment" data (where the first dimension is the number of epochs).

**Usage Example**

~~~python
from dataset.sleepedfx_dataset import SleepEDFCassetteDataset
from dataset.transform import SlideWindow, Select, PickChannels, Mapping, Transpose, Reshape, ExpandDims, Compose

before_segment_transform =[
    PickChannels(channels=['Fpz-Cz'], source='eeg', target='eeg'),
]
offline_signal_transform = [
    SlideWindow(window_size=20, stride=20, axis=0, source='eeg', target='eeg'),
    SlideWindow(window_size=20, stride=20, axis=0, source='eog', target='eog'),
    Select(key=['eeg', 'eog']),
]
offline_label_transform = [
    Mapping(
        mapping={
            0:0,  # Sleep stage W
            1:1,  # Sleep stage N1
            2:2,  # Sleep stage N2
            3:3,  # Sleep stage N3
            4:3, # Sleep stage N4
            5:4,  # Sleep stage R
        },source='stage', target='stage'),
    SlideWindow(window_size=20, stride=20, axis=0, source='stage', target='stage'),
]
dataset = SleepEDFCassetteDataset(
    root_path='/mnt/ssd/lingyus/sleep-edf-20',
    # root_path='/mnt/ssd/lingyus/test',
    io_path='/mnt/ssd/lingyus/tyee_sleepedfx_20/train',
    io_mode='hdf5',
    before_segment_transform=before_segment_transform,
    offline_signal_transform=offline_signal_transform,
    offline_label_transform=offline_label_transform,
    online_signal_transform=online_signal_transform,
    io_chunks=20,
    crop_wake_mins=30,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## SleepEDFTelemetryDataset

~~~python
class SleepEDFTelemetryDataset(BaseDataset):
    def __init__(
        self,
        # Telemetry dataset parameters (same as Cassette for crop_wake_mins)
        crop_wake_mins: int = 30,
        # BaseDataset parameters
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
~~~

The `SleepEDFTelemetryDataset` class is used to handle the "sleep-telemetry" (ST) part of the Sleep-EDF Database Expanded. This sub-dataset contains polysomnography recordings using telemetry equipment, primarily for studying the effects of age on sleep. This class is very similar to `SleepEDFCassetteDataset` but handles slightly different signal channels.

**Main Initialization Parameters**

- **crop_wake_mins** (`int`): Same as `SleepEDFCassetteDataset`, used to truncate wake periods at both ends of the recording. Defaults to `30` minutes.
- **root_path** (`str`): The root directory path for the Sleep-EDF "sleep-telemetry" dataset. This directory should contain pairs of `ST*.edf` (PSG signals) and `ST*Hypnogram.edf` (sleep stage annotations) files. Defaults to `'./sleep-edf-database-expanded-1.0.0/sleep-telemetry'`.
- Other `BaseDataset` parameters such as `start_offset`, `end_offset`, `include_end` (these are typically kept at their default value of 0 in this dataset, as segmentation is mainly driven by MNE's 30-second epoch events), as well as various transform callback functions and IO parameters, have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:

  - Similar to "sleep-cassette", each recording includes one `*PSG.edf` file and a `*Hypnogram.edf` file.

- Supported signal types (loaded from 

  ```
  *PSG.edf
  ```

   files):

  - `eeg`: Electroencephalography signals (2 channels: 'Fpz-Cz', 'Pz-Oz').
  - `eog`: Electrooculography signal (1 channel: 'horizontal').
  - `emg`: Electromyography signal (1 channel: 'submental' - submental muscle).
  - **Note**: Unlike "sleep-cassette", telemetry data typically does not include respiration and body temperature signals.

- Sampling rate:

  - Read from the `.edf` file header information. EEG, EOG, EMG are typically **100 Hz**.

- Label information:

  - `'stage'`: Sleep stage labels, extracted from annotations in the `*Hypnogram.edf` file.
  - Sleep stages are typically scored in 30-second epochs.
  - Label mapping (0-indexed):
    - `0`: 'W' (Wake)
    - `1`: '1' (NREM Stage 1, N1)
    - `2`: '2' (NREM Stage 2, N2)
    - `3`: '3' (NREM Stage 3, N3)
    - `4`: '4' (NREM Stage 4, N4) (In some scoring standards, N3 and N4 are combined into N3)
    - `5`: 'R' (REM, Rapid Eye Movement)
    - Undefined or movement artifact periods may be mapped to `-1` or ignored.

- **Segmentation (specific behavior of `segment_split`)**: This class overrides the `segment_split` method. It first extracts valid data segments for each signal type based on `labels['segments']` (generated by `read_record`, where each element represents a 30-second sleep epoch). Then, it stacks **all valid 30-second data segments** separately by signal type. Therefore, unless further transformations like `SlideWindow` are applied, `BaseDataset`'s `process_record` method, when processing each raw recording file, primarily operates on this aggregated "whole segment" data (where the first dimension is the number of epochs).

**Usage Example**

~~~python
from dataset.sleepedfx_dataset import SleepEDFCassetteDataset
from dataset.transform import SlideWindow, Select, PickChannels, Mapping, Transpose, Reshape, ExpandDims, Compose

before_segment_transform =[
    PickChannels(channels=['Fpz-Cz'], source='eeg', target='eeg'),
]
offline_signal_transform = [
    SlideWindow(window_size=20, stride=20, axis=0, source='eeg', target='eeg'),
    SlideWindow(window_size=20, stride=20, axis=0, source='eog', target='eog'),
    Select(key=['eeg', 'eog']),
]
offline_label_transform = [
    Mapping(
        mapping={
            0:0,  # Sleep stage W
            1:1,  # Sleep stage N1
            2:2,  # Sleep stage N2
            3:3,  # Sleep stage N3
            4:3, # Sleep stage N4
            5:4,  # Sleep stage R
        },source='stage', target='stage'),
    SlideWindow(window_size=20, stride=20, axis=0, source='stage', target='stage'),
]
dataset = SleepEDFTelemetryDataset(
    root_path='/mnt/ssd/lingyus/sleep-edf-20',
    # root_path='/mnt/ssd/lingyus/test',
    io_path='/mnt/ssd/lingyus/tyee_sleepedfx_20/train',
    io_mode='hdf5',
    before_segment_transform=before_segment_transform,
    offline_signal_transform=offline_signal_transform,
    offline_label_transform=offline_label_transform,
    online_signal_transform=online_signal_transform,
    io_chunks=20,
    crop_wake_mins=30,
    num_worker=8,
)
~~~

[`Back to Top`](#tyeedataset)

## TUABDataset

~~~python
class TUABDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './tuh_eeg_abnormal/v3.0.1/edf/train',
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
        io_mode: str = 'hdf5',
        num_worker: int = 0,
        lazy_threshold: int = 128,
        verbose: bool = True,
    ) -> None:
~~~

The `TUABDataset` class is used to handle the TUAB (Temple University Hospital Abnormal EEG Corpus) dataset. This dataset contains a large number of clinical Electroencephalogram (EEG) recordings, with each recording labeled as "normal" or "abnormal". This class inherits from `BaseDataset` and implements logic for loading EEG signals from `.edf` files and extracting corresponding labels from the directory structure. Specifically, it typically processes EDF files that have undergone specific preprocessing (e.g., `01_tcp_ar` average reference montage).

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the TUAB dataset, e.g., pointing to a path containing subdirectories like `train` or `eval`. This class recursively searches for `.edf` files in subdirectories whose paths contain `'01_tcp_ar'`. Defaults to `'./tuh_eeg_abnormal/v3.0.1/edf/train'` (please modify according to the actual path).
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other `BaseDataset` parameters such as `start_offset`, `end_offset`, `include_end` (these are typically kept at their default value of 0 in this dataset, as the entire recording is considered an initial segment, and subsequent segmentation is usually done through transformations like `SlideWindow`), as well as various transform callback functions and IO parameters, have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Signal data is stored in `.edf` files. The `set_records` method recursively searches under `root_path` for `.edf` files located in directories whose paths contain the string `'01_tcp_ar'`.
- Supported signal types:
  - `eeg`: Electroencephalography signals.
- Sampling rate:
  - **256Hz**, dynamically read from the header information of each `.edf` file (`Rawdata.info['sfreq']`).
- Channel information:
  - Original channel names are read from the `.edf` file.
  - Channel names are standardized (e.g., simplified from `EEG FP1-REF` to `FP1`).
- Label information:
  - `'label'`: Binary label, indicating whether the EEG recording is "normal" or "abnormal".

**Usage Example**

~~~python
from dataset.tuab_dataset import TUABDataset
from dataset.transform import SlideWindow
from dataset.transform import Resample, Compose, Filter, NotchFilter,PickChannels
from dataset.transform import Select

chanels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
offline_signal_transform = [
    SlideWindow(window_size=2000,stride=2000,source='eeg', target='eeg'),
]
before_segment_transform = [
    Compose([
        PickChannels(channels=chanels),
        Filter(l_freq=0.1, h_freq=75.0),
        NotchFilter(freqs=[50.0]),
        Resample(desired_freq=200),
        ], source='eeg', target='eeg'
    ),
]
offline_label_transform = [
    Mapping(mapping={
        'abnormal': 1,
        'normal': 0,
    }, source='label', target='label'),
    Select(key='label')
]

dataset = TUABDataset(
    root_path='/mnt/ssd/lingyus/test',
    io_path='/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/processed_train',
    before_segment_transform=before_segment_transform,
    offline_signal_transform=offline_signal_transform,
    io_mode='hdf5',
    num_worker=8
)
~~~

[`Back to Top`](#tyeedataset)

## TUEVDataset

~~~python
class TUEVDataset(BaseDataset):
    def __init__(
        self,
        root_path: str = './tuh_eeg_events/v2.0.1/edf/train',
        start_offset: float = -2.0,
        end_offset: float = 2.0,
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
~~~

Okay, here's the English translation of the `TUEVDataset` class description:

------

The `TUEVDataset` class is used to handle the TUH EEG Events (TUEV) Corpus dataset. This dataset contains a large number of clinical Electroencephalogram (EEG) recordings, annotated with various types of EEG events. This class inherits from `BaseDataset` and implements logic for loading EEG signals from `.edf` files and event annotations from corresponding `.rec` (event recording) files.

**An important data processing characteristic**: In the `process_record` method, this class **triplicates and concatenates** the raw EEG signal data along the time axis (`data = np.concatenate([data, data, data], axis=1)`). Subsequently, an offset equal to the length of the original (single) signal is added to the event timestamps read from the `.rec` file. This means the event timestamps are effectively positioned relative to the middle section of this tripled-length signal. The `segment_split` method of `BaseDataset` then uses the `start_offset` and `end_offset` parameters (relative to these adjusted event timestamps) to extract data segments.

**Main Initialization Parameters**

- **root_path** (`str`): The root directory path for the TUH EEG Events dataset. This directory should recursively contain `.edf` files and their corresponding `.rec` event files. Defaults to `'./tuh_eeg_events/v2.0.1/edf/train'` (please modify according to the actual path).
- **start_offset** (`float`): The start time (in seconds) relative to each (adjusted) event marker, used for segment extraction. Defaults to `-2.0` seconds (i.e., 2 seconds before the event).
- **end_offset** (`float`): The end time (in seconds) relative to each (adjusted) event marker, used for segment extraction. Defaults to `2.0` seconds (i.e., 2 seconds after the event).
- **include_end** (`bool`): Whether to include the sample corresponding to the end time point (calculated from `end_offset`) when segmenting. Defaults to `False`.
- **io_path** (`Union[None, str]`): Path for caching preprocessed data.
- Other `BaseDataset` parameters such as `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` have the same functionality as in `BaseDataset`.

**Dataset Characteristics**

- Raw data files:
  - Signal data is stored in `.edf` files.
  - Event annotations (type and time) are stored in text files with the same name as the `.edf` file but with a `.rec` extension. The `set_records` method recursively finds all `.edf` files under `root_path`.
- Supported signal types:
  - `eeg`: Electroencephalography signals.
- Sampling rate:
  - **256Hz**, dynamically read from the header information of each `.edf` file (`Rawdata.info['sfreq']`).
- Channel information:
  - Original channel names are read from the `.edf` file; channel names are standardized (e.g., simplified from `EEG FP1-REF` to `FP1`).
- Label information:
  - `'event'`: EEG event type labels, read from the `.rec` file.

**Usage Example**

~~~python
from dataset.tuev_dataset import TUEVDataset
from dataset.transform import SlideWindow
from dataset.transform import Resample, Compose, Filter, NotchFilter,PickChannels, Offset
from dataset.transform import Select

chanels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
offline_signal_transform = [
    SlideWindow(window_size=1000,stride=1000,source='eeg', target='eeg'),
]
before_segment_transform = [
    Compose([
        PickChannels(channels=chanels),
        Filter(l_freq=0.1, h_freq=75.0),
        NotchFilter(freqs=[50.0]),
        Resample(desired_freq=200),
        ], source='eeg', target='eeg'
    ),
]
offline_label_transform = [
    Offset(offest=-1,source='event', target='event'),
    Select(key='event')
]
dataset = TUEVDataset(
    root_path='/mnt/ssd/lingyus/tuev_test',
    io_path='/mnt/ssd/lingyus/tuev_test/processed',
    before_segment_transform=before_segment_transform,
    offline_signal_transform=offline_signal_transform,
    offline_label_transform=offline_label_transform,
    num_worker=8
)
~~~
[`Back to Top`](#tyeedataset)
