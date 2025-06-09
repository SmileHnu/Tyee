# tyee.dataset
## 目录

- [如何构建自定义数据集 (继承 `BaseDataset`)](#如何构建自定义数据集-继承-basedataset)

| 数据集类名称                                                 | 功能描述                                                     | 数据类型           | 采样率          | 应用领域                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | --------------- | ------------------------ |
| [`BaseDataset`](#basedataset)                               | 所有数据集类的基类，提供数据集初始化、预处理、缓存和加载等核心功能 | 通用               | 可配置          | 基础框架                 |
| [`BCICIV2ADataset`](#bciciv2adataset)                       | 处理BCI Competition IV Dataset 2a数据集，用于运动想象任务分类 | EEG, EOG           | 250 Hz          | 脑机接口                 |
| [`BCICIV4Dataset`](#bciciv4dataset)                         | 处理BCI Competition IV Dataset 4的ECoG数据，用于手指屈曲预测 | ECoG, Dataglove    | 1000 Hz         | 脑机接口                 |
| [`CinC2018Dataset`](#cinc2018dataset)                       | 处理PhysioNet/CinC Challenge 2018数据集，用于睡眠阶段自动分类 | EEG, EOG, EMG等    | 200 Hz          | 睡眠分析                 |
| [`DaLiADataset`](#daliadataset)                             | 处理PPG_DaLiA数据集，用于基于PPG信号的心率估计              | PPG, ACC           | 64 Hz, 32 Hz    | 心率监测                 |
| [`DEAPDataset`](#deapdataset)                               | 处理DEAP数据集，用于基于多模态生理信号的情感识别             | EEG, EOG, EMG等    | 128 Hz          | 情感计算                 |
| [`KaggleERNDataset`](#kaggleerndataset)                     | 处理Kaggle "Grasp-and-Lift EEG Detection"数据集，用于错误相关负电位检测 | EEG                | 200 Hz          | 脑机接口                 |
| [`MITBIHDataset`](#mitbihdataset)                           | 处理MIT-BIH心律失常数据库，用于心电图心律失常检测            | ECG                | 360 Hz          | 心电分析                 |
| [`NinaproDB5Dataset`](#ninaprodb5dataset)                   | 处理Ninapro DB5数据集，用于基于sEMG信号的手势识别           | sEMG               | 200 Hz          | 手势识别                 |
| [`PhysioP300Dataset`](#physiop300dataset)                   | 处理PhysioNet P300数据集，用于P300拼写器等脑机接口应用      | EEG                | 2048 Hz         | 脑机接口                 |
| [`SEEDVFeatureDataset`](#seedvfeaturedataset)               | 处理SEED-V数据集的预提取特征，用于多模态情感识别             | EEG特征, 眼动特征  | 特征数据        | 情感计算                 |
| [`SleepEDFCassetteDataset`](#sleepedfcassetedataset)        | 处理Sleep-EDF Database Expanded中的cassette部分，用于睡眠阶段分析 | EEG, EOG, EMG等    | 100 Hz, 1 Hz    | 睡眠分析                 |
| [`SleepEDFTelemetryDataset`](#sleepedftelemetrydataset)     | 处理Sleep-EDF Database Expanded中的telemetry部分，用于睡眠阶段分析 | EEG, EOG, EMG      | 100 Hz          | 睡眠分析                 |
| [`TUABDataset`](#tuabdataset)                               | 处理TUAB数据集，用于临床EEG异常检测                         | EEG                | 256 Hz          | 临床诊断                 |
| [`TUEVDataset`](#tuevdataset)                               | 处理TUH EEG Events数据集，用于EEG事件检测                   | EEG                | 256 Hz          | 临床诊断                 |

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

`BaseDataset` 是用于管理各类生理信号数据集的基类。它提供了数据集初始化、原始数据处理、数据读取与写入、元数据管理、懒加载/即时加载模式切换以及应用预处理变换（“钩子函数”）等核心功能。当 `io_path` 指定的缓存路径为空或 `io_mode` 为 `'memory'` 时，会自动进行数据预处理并将结果写入 `io_path`；否则，将直接从缓存加载数据。

**核心设计理念：**

`BaseDataset` 的设计旨在将原始数据的读取、处理、分段、存储与最终的数据获取解耦。开发者通过继承此类并实现特定的抽象方法，可以快速构建针对不同原始数据集格式的处理流程。处理后的数据和元信息会被高效地存储（例如使用HDF5或LMDB），以供后续的训练和分析。

**主要初始化参数**

- **root_path** (`str`): 原始数据集所在的根目录路径。
- **io_path** (`Union[None, str]`): 用于存储/读取预处理结果（缓存）的路径。如果为 `None` 或路径下无有效缓存，将触发数据预处理流程。
- **start_offset** (`float`): 在从原始信号中提取分段时，应用于每个分段起始时间的偏移量（秒）。默认为 `0.0`。
- **end_offset** (`float`): 应用于每个分段结束时间的偏移量（秒）。默认为 `0.0`。
- **include_end** (`bool`): 在按时间分段时，是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False` (即不包含)。
- **before_segment_transform** (`Union[None, Callable]`): 一个可选的变换函数（或变换列表），在数据分段（调用 `segment_split`）之前，应用于从 `read_record` 读取的完整记录数据中的 `signals` 部分。
- **offline_signal_transform** (`Union[None, Callable]`): 一个可选的变换函数（或变换列表），在数据处理阶段（`process_record`内部，通常是对每个分段后）应用于信号数据。变换结果会被缓存。
- **offline_label_transform** (`Union[None, Callable]`): 一个可选的变换函数（或变换列表），在数据处理阶段应用于标签数据。变换结果会被缓存。
- **online_signal_transform** (`Union[None, Callable]`): 一个可选的变换函数（或变换列表），在数据加载阶段（`__getitem__`中）动态应用于读取出的信号数据。变换结果不会被缓存。
- **online_label_transform** (`Union[None, Callable]`): 一个可选的变换函数（或变换列表），在数据加载阶段动态应用于读取出的标签数据。变换结果不会被缓存。
- **io_mode** (`str`): 预处理结果的存储模式，例如 'hdf5', 'lmdb', 'pickle', 'memory'。默认为 `'hdf5'`。
- **io_size** (`int`): 数据库的最大容量（字节）。默认为 `1048576` (1MB)。
- **io_chunks** (`int`, optional): HDF5存储模式下的数据分块大小。
- **num_worker** (`int`): 数据预处理时使用的并行工作进程数。`0` 表示单进程。默认为 `0`。
- **lazy_threshold** (`int`): 当记录（record）数量超过此阈值时，数据集会切换到懒加载模式（仅在需要时加载IO处理器）。默认为 `128`。
- **verbose** (`bool`): 是否在数据预处理时显示进度条。默认为 `True`。
- ***\*kwargs**: 其他传递给具体数据集实现中各个方法的参数（如 `set_records`, `read_record`, `process_record`）。

[`返回目录`](#tyeedataset)

## 如何构建自定义数据集 (继承 `BaseDataset`)

要基于 `BaseDataset` 构建针对特定原始数据集的处理流程，您需要创建一个新的类继承自 `BaseDataset`，并至少实现以下三个核心的抽象方法：`set_records`, `read_record`, 和 `process_record`。

### 1. 必须实现的方法

以下方法在您的子类中**必须被重写**：

- `set_records(self, root_path: str, \**kwargs) -> List[Any]`
  - **功能**: 定义需要被处理的原始数据记录单元列表。这些记录单元可以是文件名、文件路径元组或任何能够被 `read_record` 方法理解并用于定位原始数据的标识符。
  - **参数:**
    - `root_path` (`str`): 通常是原始数据存放的根目录，在 `BaseDataset` 的 `__init__` 中传入。
    - `**kwargs`: 从 `BaseDataset` 的 `__init__` 传递过来的其他参数。
  - **返回**: `List[Any]` - 一个列表，其中每个元素代表一个待处理的记录。
  - **示例**

```python
def set_records(self, root_path: str, **kwargs) -> List[str]:
    # 假设原始数据是根目录下的 .edf 文件
    records = []
    for filename in os.listdir(root_path):
        if filename.endswith('.edf'):
            records.append(os.path.join(root_path, filename))
    return records
```

- `read_record(self, record: Any, \**kwargs) -> Dict`
  - **功能**: 接收一个由 `set_records` 返回的记录标识符，负责读取该原始记录的所有相关数据，包括各种类型的信号、标签以及元数据。
  - **参数**:
    - `record` (`Any`): `set_records` 返回列表中的一个元素。
    - `**kwargs`: 从 `BaseDataset` 的 `__init__` 传递过来的其他参数。
  - **返回**: `Dict`- 一个包含读取数据的字典，其结构应符合以下约定：
    - 顶层键 `'signals'` (必需): 其值为一个字典，其中每个键是信号类型名 (例如 'eeg', 'ecg')，对应的值是另一个字典，包含:
      - `'data': np.ndarray`: 信号数据 (例如 `(通道数, 采样点数)` 数组)。
      - `'channels': List[str]`: 通道名称列表。
      - `'freq': float`: 该信号的采样频率。
    - 顶层键 `'labels'`(必需): 其值为一个字典，描述了与信号相关的标签。通常包含一个 `'segments'`键，其值为一个列表，列表中的每个元素是一个定义了标注段的字典：
      - `'start': float`: 分段的开始时间 (单位：秒)。
      - `'end': float`: 分段的结束时间 (单位：秒)。
      - `'value': Dict`: 一个字典，包含了这个时间段内所有类型的真实标签。该字典的键是标签类型名 (例如 'sleep_stage', 'event_type')，值是包含实际标签数据的字典，如 `{'data': label_value}`。
    - 顶层键 `'meta'` (必需): 其值为一个字典，包含与此记录相关的任何其他元数据 (例如原始文件名、被试ID等)。
  - **示例返回结构**:

```python
{
    'signals': {
        'eeg': {'data': eeg_data_array, 'channels': ['C3', 'C4'], 'freq': 250.0},
        'ecg': {'data': ecg_data_array, 'channels': ['ECG'], 'freq': 500.0}
    },
    'labels': {
        'segments': [ # 如果是事件/分段标签
            {'start': 0.5, 'end': 2.3, 'value': {'sleep_stage': {'data': 1}, 'arousal': {'data': 0}}},
            {'start': 5.0, 'end': 10.0, 'value': {'sleep_stage': {'data': 2}}}
        ]
    'meta': {
        'file_name': str(record), # 假设 record 是文件名
        # ... 其他元数据 ...
    }
}
```

- `process_record(self, signals: Dict, labels: Dict, meta: Dict, **kwargs) -> Generator[Dict, None, None]`

  - **功能**: 此方法是一个**生成器**。它接收从 `read_record` 返回并解包后的 `signals`, `labels`, `meta` 字典（以及其他来自 `__init__` 的 `kwargs`），负责对这些数据进行核心处理。
  - **典型工作流程**:
    1. （可选）对 `signals` 应用 `before_segment_transform`。
    2. 调用 `self.segment_split(processed_signals, labels)` 将数据按时间分段。
    3. 遍历每个分段:
       - 获取分段的信号 (`seg_signals`)、标签 (`seg_labels`) 和时段信息 (`seg_info`)。
       - 为该分段生成一个唯一的 `segment_id` (例如使用 `self.get_segment_id(meta['original_filename'], segment_index)`)。
       - （可选）对 `seg_signals` 应用 `offline_signal_transform`。
       - （可选）对 `seg_labels` 应用 `offline_label_transform`。
       - **填充 `seg_info`**: 使用 `meta` 信息和 `get_subject_id` 等辅助方法，向 `seg_info` 中添加如 `subject_id`, `session_id`, `trial_id`, `segment_id`  等元数据。
       - 调用 `self.assemble_segment(key=segment_id, signals=seg_signals, labels=seg_labels, info=seg_info)` 构建最终的输出字典。
       - `yield` 这个由 `assemble_segment` 返回的字典。
  - **参数**:
    - `signals` (`Dict`): 即 `read_record` 返回字典中的 `'signals'` 部分。
    - `labels` (`Dict`): 即 `read_record` 返回字典中的 `'labels'` 部分。
    - `meta` (`Dict`): 即 `read_record` 返回字典中的 `'meta'` 部分。
    - `**kwargs`: 其他从 `BaseDataset` 的 `__init__` 和 `handle_record` 传递过来的参数。
  - `yield` 的字典结构(**无需用户定义**）: 每个产出的字典代表一个将要被存储的数据单元（通常是一个分段），其结构由 `assemble_segment`方法确定，并需要符合 `BaseDataset`内部写入方法 (`write_signal`, `write_label`, `write_info`) 的期望。大致结构如下：
    - `'key': str`：用于存储的唯一分段ID。
    - `'signals': Dict[str, Dict]` (可选)：包含处理后信号的字典。每个内部信号字典包含 `'data'`, `'channels'`, `'freq'` 及一个 `'info'` 字典（含 `'sample_ids'`, `'windows'`）。
    - `'labels': Dict[str, Dict]` (可选)：包含处理后标签的字典。每个内部标签字典包含 `'data'` 及一个 `'info'` 字典（含 `'sample_ids'`, 可选的 `'windows'`）。
    - `'info': Dict` (必需)：包含此数据段的元数据，供 `MetaInfoIO` 写入。**必须包含 `'sample_ids': List[str]`**，以及其他如 `subject_id`, `session_id` 等。
  - 示例逻辑框架

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

### 2. 可选重写的方法

以下方法提供了默认实现，但您可以根据具体的数据集特性和处理需求进行重写：

- **`segment_split(self, signals: Dict, labels: Dict) -> List[Dict]`**:
  - 默认实现是根据 `labels['segments']` 列表中的 `start` 和 `end` 时间（秒）来切分 `signals` 中的所有信号。`start_offset` 和 `end_offset` 会被应用。
  - 如果您的数据分段逻辑不同（例如，固定长度的滑动窗口，或标签并非以时间段形式给出），您需要重写此方法。
  - 它应返回一个列表，每个元素是一个字典，代表一个分段（包含分段后的 `'signals'`, `'labels'` 和时段 `'info'`）。
- **`get_subject_id(\**kwargs) -> str`**, **`get_session_id(\**kwargs) -> str`**, **`get_trial_id(\**kwargs) -> str`** ,**`get_segment_id(\**kwargs)->str`**:
  - 这些静态方法（或实例方法，取决于您的设计）用于从传递给 `process_record` 的参数中（例如从 `meta` 字典或 `record` 标识符，或者像您示例中那样从文件名和索引）提取被试ID、会话ID、试验ID等。
  - `BaseDataset` 中默认的这些方法均返回 `"0"`。如果您的数据包含这些层级，您**必须**重写它们以正确提取这些ID，以便在 `process_record` 中构建 `info` 字典时使用。

### 3. 通常不应重写的方法 (内部辅助方法)

- **`assemble_segment(...) -> Dict`**:
  - 负责将处理好的单个分段的 `signals`, `labels`, `key` 和 `info` 组装成 `process_record` 最终 `yield` 的标准字典结构，特别是处理 `signals` 和 `labels` 内部的 `'info'` 字段（如 `sample_ids`, `windows`）。**此方法具有固定的输出结构以匹配数据写入机制，通常不应被用户重写。**
- **`assemble_sample(self, signals: Dict, labels: Dict) -> Dict`**:
  - 在 `__getitem__` 中被调用，负责将从磁盘读取的 `signals` 和 `labels` 组装成数据加载器 (DataLoader) 所期望的单个样本格式。默认实现是将所有信号类型的数据和标签类型的数据直接放入返回字典的顶层。**此方法也具有特定的预期行为以供 DataLoader 使用，通常不应被用户重写。**

### 4. 数据加载与在线变换

- **`__getitem__(self, index: int) -> Dict`**:
  - `BaseDataset` 已实现此方法。它根据索引从 `self.info` (通常是 `pd.DataFrame`) 中获取样本的元数据，然后使用 `read_signal` 和 `read_label` 从磁盘（或缓存）加载对应的信号和标签数据。
  - 加载数据后，会分别应用 `online_signal_transform` 和 `online_label_transform`。
  - 最后调用 `assemble_sample` 组装并返回样本。
- **`online_signal_transform` / `online_label_transform`**:
  - 这些在 `__init__` 中设置的变换在每次通过 `__getitem__` 请求数据时动态应用，适合那些针对单个样本，或是追求更灵活的处理操作。

[`返回目录`](#tyeedataset)

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

`BCICIV2ADataset` 类用于处理 BCI Competition IV Dataset 2a 数据集。它继承自 `BaseDataset`，负责读取原始的 `.gdf` 格式的脑电信号(EEG)和眼电信号(EOG)数据，以及对应的 `.mat` 格式的标签文件。该数据集通常包含多个被试，每个被试有训练（'T'）和评估（'E'）两种会话，会话中包含对应四种运动想象任务（左手、右手、脚、舌头）的试验。数据段通常根据GDF文件中的事件标记（事件ID '768'）结合指定的偏移量来提取。

**主要初始化参数**

- **root_path** (`str`): BCI Competition IV Dataset 2a 数据集的根目录路径，该目录应包含所有被试的原始数据文件（例如 `A01T.gdf`, `A01T.mat`, `A02T.gdf`, `A02T.mat` 等）。默认为 `'./BCICIV_2a'`。
- **start_offset** (`float`): 相对于事件标记的开始时间（秒），用于分段提取。默认为 `2.0` 秒。
- **end_offset** (`float`): 相对于事件标记的结束时间（秒），用于分段提取。默认为 `6.0` 秒。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。如果指定，预处理后的数据将存储在此路径，后续实例化时可直接加载。
- **offline_signal_transform**, **offline_label_transform** (`Union[None, Callable]`): 在数据预处理阶段（离线）应用的变换，其结果会被缓存。
- **online_signal_transform**, **online_label_transform** (`Union[None, Callable]`): 在数据加载阶段（在线）应用的变换，结果不会被缓存。
- 其他参数如 `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 [`BaseDataset`](##BaseDataset)。

**数据集特性**

- 支持的信号类型:
  - `eeg`: 脑电信号。
  - `eog`: 眼电信号。
- 采样率:
  - 所有信号（EEG 和 EOG）的采样率均为 **250 Hz**。
- 通道信息:
  - `eeg`: 包含22个标准的EEG通道。
  - `eog`: 包含3个EOG通道，通常处理后命名为 `['left', 'central', 'right']`。
- 标签信息:
  - 标签从 `.mat` 文件中加载，对应于运动想象任务。
  - 原始标签为 1, 2, 3, 4，在本数据集中被转换为 **0, 1, 2, 3**。
  - 类别对应关系：
    - `0`: 左手 (left hand)
    - `1`: 右手 (right hand)
    - `2`: 脚 (foot)
    - `3`: 舌头 (tongue)

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`BCICIV4Dataset` 类用于处理 BCI Competition IV Dataset 4 的 ECoG (脑皮层电图) 数据。该数据集包含多个被试，每个被试的数据文件（`.mat`格式，如 `sub1_comp.mat`）中包含了训练阶段和测试阶段的ECoG信号以及对应的手指屈曲数据（Dataglove，作为此处的标签）。此类继承自 [`BaseDataset`](##BaseDataset)，并实现了特定于此数据集的数据读取和预处理逻辑。

**主要初始化参数**

- **root_path** (`str`): BCI Competition IV Dataset 4 数据集的根目录路径，该目录应包含各个被试的 `*_comp.mat` 和 `*_testlabels.mat` 文件。默认为 `'./BCICIV4'`。
- **start_offset** (`float`): 相对于每个分段（此处指训练段和测试段）的开始时间的偏移量（秒）。默认为 `0.0` 秒。
- **end_offset** (`float`): 相对于每个分段结束时间的偏移量（秒）。默认为 `0.0` 秒。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。如果指定，预处理后的数据将存储在此路径，后续实例化时可直接加载。
- **offline_signal_transform**, **offline_label_transform** (`Union[None, List[Callable]]`): 在数据预处理阶段（离线）应用的变换列表，其结果会被缓存。
- **online_signal_transform**, **online_label_transform** (`Union[None, List[Callable]]`): 在数据加载阶段（在线）应用的变换列表，结果不会被缓存。
- 其他参数如 `io_size`, `io_chunks`, `io_mode`, `num_worker`, `verbose` 功能同 [`BaseDataset`](#basetdataset)。

**数据集特性**

- 支持的信号类型:
  - `ecog`: 脑皮层电图信号。数据从 `.mat` 文件中的 `train_data` 和 `test_data` 字段加载并拼接。
- 采样率:
  - `ecog`: **1000 Hz**。
  - `dg` (Dataglove，作为标签处理): 原始采样率为 **1000 Hz**，可在线下变换中进行重采样。
- 通道信息:
  - `ecog`: 通道在数据集中从 "0" 开始按数字索引命名。
- 标签信息 (Dataglove):
  - 标签数据是连续的多通道Dataglove手指屈曲信号。
  - 在 `read_record` 中，原始的ECoG数据被逻辑地分为“训练段”和“测试段”，分别对应于 `.mat` 文件中的 `train_data` 和 `test_data`。
  - 每个逻辑段（训练段/测试段）的Dataglove数据 (`train_dg` 或 `test_dg`) 作为其标签值，结构为 `{'dg': {'data': dataglove_array, 'freq': 1000}}`。
  - 因此，`BaseDataset`的 `segment_split` 方法（如果使用默认）会根据这两个逻辑段的时间范围来切分ECoG和对应的Dataglove数据。`start_offset` 和 `end_offset` 会应用于这两个主要分段。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`CinC2018Dataset` 类用于处理 PhysioNet/CinC Challenge 2018 数据集，该数据集专注于使用多通道生理信号进行睡眠阶段自动分类。此类继承自 `BaseDataset`，并实现了从特定文件格式（`.hea` 用于头文件，`.mat` 用于信号数据，`*-arousal.mat` 用于唤醒/睡眠阶段标签）加载和预处理数据的逻辑。

**主要初始化参数**

- **root_path** (`str`): PhysioNet Challenge 2018 训练数据集的根目录路径。该目录应包含各个记录的子文件夹，每个子文件夹包含对应的 `.hea`, `.mat` (信号) 和 `*-arousal.mat` (标签) 文件。默认为 `'./challenge-2018/training'`。
- **start_offset** (`float`): 相对于每个30秒睡眠分期事件标记的开始时间的偏移量（秒）。在此数据集中，由于标签本身定义了分期，此参数通常配合 `BaseDataset` 默认的 `segment_split` 使用，但 `CinC2018Dataset` 重写了 `segment_split`。默认为 `0.0` 秒。
- **end_offset** (`float`): 相对于每个30秒睡眠分期事件标记的结束时间的偏移量（秒）。默认为 `0.0` 秒。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:

  - `.hea` 文件: 包含记录的头信息，如信号名称、采样频率等。
  - `.mat` 文件 (例如 `tr03-0005.mat`): 包含实际的多通道生理信号数据。
  - `*-arousal.mat` 文件: 包含睡眠阶段和唤醒事件的标注。

- 支持的信号类型

   (从 `.mat`文件加载并根据 `.hea`文件信息分组):

  - `eeg`: 脑电信号 (通常为前6个通道，如 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1')。
  - `eog`: 眼电信号 (通常为第7个通道，如 'E1-M2')。
  - `emg`: 肌电信号 (通常为第8个通道，如 'Chin1-Chin2')。
  - `abd`: 腹部呼吸信号。
  - `chest`: 胸部呼吸信号。
  - `airflow`: 气流信号。
  - `sao2`: 血氧饱和度信号。
  - `ecg`: 心电信号。

- 采样率:

  - 从每个记录的 `.hea` 文件中动态读取 (通过 `import_signal_names` 方法)。对于此数据集，通常为 **200 Hz**。

- 通道信息:

  - 从 `.hea` 文件中读取。具体通道名称和数量可能因记录而异，但通常遵循上述信号类型分组。

- 标签信息:

  - 从 `*-arousal.mat` 文件中提取睡眠阶段标签。
  - 标签被处理成30秒为单位的睡眠分期。
  - 睡眠阶段映射关系
    - "wake": 0
    - "nonrem1": 1
    - "nonrem2": 2
    - "nonrem3": 3
    - "rem": 4

- **分段 (`segment_split` 的特定行为)**: 此类重写了 `segment_split` 方法。它会聚合一个记录文件内所有有效的30秒睡眠分期数据，然后按信号类型和标签类型分别堆叠。因此，除非应用了如 `SlideWindow` 等进一步的变换，否则 `BaseDataset` 的 `process_record` 在处理每个原始记录文件时，主要针对的是这个聚合后的“整段”数据（其中第一维是分期数量）。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`DaLiADataset` 类用于PPG_DaLiA 数据集加载和处理生理信号，如光电容积描记(PPG)和加速度计(ACC)数据，以及相应的从 `.pkl` 文件中获取的心率(HR)标签。此类继承自 `BaseDataset`。

**主要初始化参数**

- **root_path** (`str`): PPG_DaLiA 数据集的根目录路径。此目录应包含各个被试的数据文件（通常为 `S*.pkl`）。默认为 `'./PPG_FieldStudy'`。
- **start_offset** (`float`): 应用于分段起始时间的偏移量（秒）。鉴于DaLiA数据最初将整个记录视为一个分段，此偏移量（如果非零）将应用于整个记录的开始。默认为 `0`。
- **end_offset** (`float`): 应用于分段结束时间的偏移量（秒）。默认为 `0`。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 支持的信号类型:
  - `ppg`: 光电容积描记信号，通常来自腕戴设备。
  - `acc`: 加速度计信号，通常来自腕戴设备（一般为3轴）。
- 采样率:
  - `ppg`: **64 Hz**。
  - `acc`: **32 Hz**。
- 标签信息:
  - `hr`: 心率数据，每个hr数据是步长为2的8秒窗口数据对应的。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`DEAPDataset` 类用于处理DEAP (Database for Emotion Analysis using Physiological Signals) 数据集。该数据集包含了32位参与者在观看40个音乐视频（每个视频时长1分钟）时的多种生理信号记录，以及他们对每个视频的情感（Valence, Arousal, Dominance, Liking）评分。此类继承自 `BaseDataset`，并负责从预处理后的Python pickled文件 (`.dat`) 中加载数据。

**主要初始化参数**

- **root_path** (`str`): DEAP数据集 `data_preprocessed_python` 文件夹的路径，该文件夹应包含所有被试的 `.dat` 文件 (例如 `s01.dat`, `s02.dat`, ..., `s32.dat`)。默认为 `'./DEAP/data_preprocessed_python'`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。注意：DEAP数据集的预处理版本已经经过了降采样、滤波等处理，`start_offset` 和 `end_offset` 在此数据集中通常保持默认值0，因为数据本身已经是分段的试验。

**数据集特性**

- 原始数据文件:
  - 每个被试的数据存储在一个 `.dat` 文件中，这是一个Python pickled对象，包含了该被试观看40个视频（试验）时的所有生理信号和情感评分。
- 支持的信号类型 (从 .dat文件中加载):
  - `eeg`: 脑电信号 (32通道)。
  - `eog`: 眼电信号 (2通道: hEOG, vEOG)。
  - `emg`: 肌电信号 (2通道: zEMG, tEMG - 分别对应颧大肌和斜方肌)。
  - `gsr`: 电皮肤活动 (Galvanic Skin Response)。
  - `resp`: 呼吸信号。
  - `ppg`: 光电容积脉搏波 (Plethysmograph)。
  - `temp`: 皮肤温度。
- 采样率:
  - 所有信号的采样率均为 **128 Hz** (这是DEAP数据集预处理版本中提供的采样率)。
- 通道信息:
  - `eeg`: 32个EEG通道，遵循国际10-20系统排列（具体顺序为FP1, AF3, F3, F7, FC5, FC1, C3, T7, CP5, CP1, P3, P7, PO3, O1, OZ, PZ, FP2, AF4, FZ, F4, F8, FC6, FC2, CZ, C4, T8, CP6, CP2, P4, P8, PO4, O2）。
  - `eog`: 2通道 (`hEOG`, `vEOG`)。
  - `emg`: 2通道 (`zEMG`, `tEMG`)。
  - `gsr`, `resp`, `ppg`, `temp`: 各为单通道信号。
- 标签信息:
  - 每个试验（视频）对应一组情感评分，范围从1到9。
  - `valence`: 效价（愉悦度）。
  - `arousal`: 唤醒度。
  - `dominance`: 控制度（优势度）。
  - `liking`: 喜爱度。
- **分段 (`segment_split` 的行为)**: DEAP数据集的原始数据已经是按试验（每个视频）分段的（每个试验60秒有效数据）。`read_record` 方法将每个试验视为一个独立的记录单元。`segment_split` 方法（在此类中被重写）会处理这些预分段的试验数据，它将每个试验（trial）的数据视为一个分段，并提取相应的信号和标签。因此，除非应用如 `SlideWindow` 这样的离线变换来进一步细分这60秒的试验数据，否则 `process_record` 将针对每个完整的60秒试验数据块进行迭代。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`KaggleERNDataset` 类用于处理来自 Kaggle "Grasp-and-Lift EEG Detection" 竞赛的数据集。该数据集专注于检测与错误相关的负电位 (Error-Related Negativity, ERN)。此类继承自 `BaseDataset`，并实现了从特定CSV文件格式加载和预处理EEG信号及对应标签的逻辑。

**主要初始化参数**

- **root_path** (`str`): Kaggle ERN 数据集的根目录路径。该目录应包含各个被试的会话数据CSV文件（例如 `Data_S02_Sess01.csv`）以及对应的标签文件 (`TrainLabels.csv` 或 `true_labels.csv`）。
- **start_offset** (`float`): 相对于反馈事件 (`FeedBackEvent`) 的开始时间（秒），用于分段提取。默认为 `-0.7` 秒 (即事件前0.7秒)。
- **end_offset** (`float`): 相对于反馈事件的结束时间（秒），用于分段提取。默认为 `1.3` 秒 (即事件后1.3秒)。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 信号数据存储在以 `Data_SXX_SessYY.csv` 格式命名的CSV文件中。
  - 标签数据根据被试ID（训练集或测试集）分别从 `TrainLabels.csv` 或 `true_labels.csv` 中加载(表格需要放到对应文件夹中）。
- 支持的信号类型:
  - `eeg`: 脑电信号。
- 采样率:
  - `eeg`: **200 Hz** 。
- 通道信息:
  - `eeg`: 包含58个EEG通道，具体通道名称列表在类中硬编码 (FP1, FP2, AF7, ..., O1, O2)。
- 标签信息:
  - 二分类标签（0 或 1），表示任务反馈是正确 (`0`) 还是错误 (`1`)。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`MITBIHDataset` 类用于处理 MIT-BIH Arrhythmia Database。该数据集包含双通道的心电图(ECG)记录，并带有详细的心律失常类型标注。此类继承自 `BaseDataset`，并实现了从WFDB格式文件加载和预处理数据的逻辑。数据段通常是围绕每个检测到的R波峰值，根据指定的偏移量来提取的。

**主要初始化参数**

- **root_path** (`str`): MIT-BIH Arrhythmia Database 数据集的根目录路径。该目录应包含WFDB格式的文件（例如 `100.dat`, `100.hea`, `100.atr` 等）。默认为 `'./mit-bih-arrhythmia-database-1.0.0'`。
- **start_offset** (`float`): 相对于每个R波峰值（作为事件标记）的开始时间（秒），用于分段提取。默认为 `-64/360` 秒（约-0.178秒），旨在捕获R波前的信号。
- **end_offset** (`float`): 相对于每个R波峰值的结束时间（秒），用于分段提取。默认为 `64/360` 秒（约+0.178秒）。结合默认的 `start_offset`，这通常会形成一个围绕R波峰值，长度为128个采样点（在360Hz采样率下）的窗口。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 数据采用WFDB格式，每个记录通常包含 `.dat` (信号), `.hea` (头文件), 和 `.atr` (注释) 文件。`root_path` 应指向包含这些记录文件的顶层目录。
- 支持的信号类型:
  - `ecg`: 心电图信号。
- 采样率:
  - 从每个记录的 `.hea` 文件中动态读取。对于MIT-BIH数据集，通常为 **360 Hz**。
- 通道信息:
  - 通常为2通道ECG信号。常见的通道包括 'MLII' (导联II的修改版) 和 'V1', 'V2', 'V4', 或 'V5' 中的一个。通道名称从 `.hea` 文件中读取。
- 标签信息:
  - 心律失常类型标注（符号）从 `.atr` 文件中读取，每个标签（符号）与一个R波峰值的位置相关联，标签是该R波峰值对应的注释符号。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`NinaproDB5Dataset` 类用于处理 Ninapro DB5 数据集。该数据集包含10位健全被试在执行约50种不同手部和腕部运动时的表面肌电（sEMG）信号。此类继承自 `BaseDataset`，并实现了从 `.mat` 文件加载数据、根据滑动窗口参数生成数据段及对应手势标签的逻辑。

**主要初始化参数**

- **wLen** (`float`): 滑动窗口的长度（单位：秒）。默认为 `0.25` 秒。
- **stepLen** (`float`): 滑动窗口的步长（单位：秒）。默认为 `0.05` 秒。
- **balance** (`bool`): 是否对生成的数据窗口进行类别平衡处理（主要针对静息状态 '0' 手势）。如果为 `True`，会尝试减少静息状态窗口的数量，使其与其他手势窗口数量大致相当。默认为 `True`。
- **include_transitions** (`bool`): 在处理标签时，是否包含手势之间的过渡窗口。如果为 `True`，包含两个手势的窗口会被特殊处理；如果为 `False`，则通常取窗口起始时的手势作为该窗口的标签。默认为 `False`。
- **root_path** (`str`): Ninapro DB5 数据集的根目录路径。该目录应递归包含各个被试的 `.mat` 文件 (例如 `S1_A1_E1.mat`, `S1_A1_E2.mat` 等)。默认为 `'./NinaProDB5'`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他 `BaseDataset` 参数如 `start_offset`, `end_offset`, `include_end`（这些通常在此数据集中保持默认值0，因为分段主要由 `wLen` 和 `stepLen` 控制），以及各种变换回调函数和IO参数，功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 每个被试的每个练习（exercise）数据存储在一个 `.mat` 文件中。`set_records` 方法会递归查找 `root_path` 下的所有 `.mat` 文件。
- 支持的信号类型:
  - `emg`: 表面肌电信号。
- 采样率:
  - 从每个 `.mat` 文件中的 'frequency' 字段动态读取。对于 Ninapro DB5，通常为 **200 Hz**。
- 通道信息:
  - EMG 通道数量从数据中获取（`emg_data.shape[0]`）。Ninapro DB5 通常包含16个EMG通道，这些通道在本类中从 "1" 开始按数字索引命名。
- 标签信息:
  - `'gesture'`: 手势标签，从 `.mat` 文件中的 'restimulus' 字段获取。
  - 原始标签代表不同的手势或静息状态。`read_record` 方法会根据文件名中的运动类别（Exercise A, B, C 对应不同的手势集）对原始标签值进行调整，以生成一个在整个数据集中唯一的手势ID。静息状态通常标记为0。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`PhysioP300Dataset` 类用于处理来自 PhysioNet 的 "ERP-based Brain-Computer Interface Recordings" 数据集，该数据集常用于P300拼写器等脑机接口研究。此类继承自 `BaseDataset`，并实现了从 `.edf` 文件加载EEG信号及事件标注，并提取与刺激相关的脑电分段及对应标签（目标/非目标刺激）的逻辑。

**主要初始化参数**

- **root_path** (`str`): PhysioNet P300 数据集的根目录路径。该目录应递归包含各个被试和会话的 `.edf` 文件。默认为 `'./lingyus/erp-based-brain-computer-interface-recordings-1.0.0'` (请根据实际路径修改)。
- **start_offset** (`float`): 相对于每个刺激事件标记的开始时间（秒），用于分段提取。默认为 `-0.7` 秒 (即事件前0.7秒)。
- **end_offset** (`float`): 相对于每个刺激事件标记的结束时间（秒），用于分段提取。默认为 `1.3` 秒 (即事件后1.3秒)。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 信号数据和事件标注存储在 `.edf` 文件中。`set_records` 方法会递归查找 `root_path` 下的所有 `.edf` 文件。
- 支持的信号类型:
  - `eeg`: 脑电信号。
- 采样率:
  - 从每个 `.edf` 文件的头信息中动态读取 (`raw.info['sfreq']`)。该数据集的原始采样率通常为 **2048 Hz**
- 通道信息:
  - 从 `.edf` 文件中读取。通常包含64个EEG通道（例如 Fp1, AF7, ..., O2 等，通道名被转换为大写）。
- 标签信息:
  - 二分类标签（0 或 1），指示呈现的刺激是否为P300任务中的目标刺激。
    - `1`: 目标刺激 (Target stimulus)
    - `0`: 非目标刺激 (Non-target stimulus)

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`SEEDVFeatureDataset` 类用于处理 SEED-V 数据集中预先提取的特征数据。SEED-V 是一个用于基于多模态生理信号（EEG和眼动数据）进行情绪识别的数据集。此类继承自 `BaseDataset`，并实现了从 `.npz` 文件加载EEG的差分熵（DE）特征和眼动特征，以及对应的情绪标签。

**主要初始化参数**

- **root_path** (`str`): SEED-V 特征数据集的根目录路径。该目录应包含 `EEG_DE_features` 和 `Eye_movement_features` 两个子目录，其中分别存放对应特征的 `.npz` 文件。默认为 `'./SEED-V'`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径（尽管此类主要加载已提取的特征，但 `BaseDataset` 框架仍使用此参数）。
- 其他 `BaseDataset` 参数如 `start_offset`, `end_offset`, `include_end`（这些在此数据集中通常保持默认值0，因为数据是特征序列且分段逻辑已在此类中特定实现），以及各种变换回调函数和IO参数，功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - EEG DE特征和眼动特征分别存储在 `EEG_DE_features` 和 `Eye_movement_features` 目录下的 `.npz` 文件中。每个 `.npz` 文件通常对应一个被试的一次实验（session）数据，内部可能包含多个试验（trial）的特征。
  - `set_records` 方法会匹配这两个目录中文件名（除去实验日期等后缀）相同（例如，同为某个被试）的 `.npz` 文件对。
- 支持的特征类型:
  - `eeg`: 脑电信号的差分熵（DE）特征。
  - `eog`: 眼动特征。
  - 注意：这些是特征向量，不是原始时域信号。
- 采样率:
  - 对于特征数据，传统意义上的“采样率”不直接适用。特征通常是从原始信号的特定时间窗口提取的。此类在 `read_record` 中未给这些特征数据明确设置 `'freq'` 字段。
- 标签信息:
  - `'emotion'`: 情绪标签，从EEG特征文件的 `label` 字段中加载。
  - SEED-V 数据集的情绪标签通常为整数，代表不同的情绪类别（例如，0: neutral, 1: sad, 2: fear, 3: happy）。
- `segment_split` 方法被**重写**：它使用 `read_record` 中为 `labels['segments']` 计算好的、基于特征向量索引的 `start` 和 `end` 来切分拼接后的整体特征数据。这意味着 `process_record` 最终处理的每个“分段”对应于原始数据中的一个试验/视频片段的特征集。

**使用举例**

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

[`返回目录`](#tyeedataset)

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

`SleepEDFCassetteDataset` 类用于处理 Sleep-EDF Database Expanded 中的 "sleep-cassette" 部分。该数据集包含从健康被试中记录的多导睡眠图(PSG)，用于睡眠阶段分析。此类继承自 `BaseDataset`，并实现了从 `.edf` 文件加载信号和睡眠阶段标注的逻辑。

**主要初始化参数**

- **crop_wake_mins** (`int`): 如果大于0，则在处理原始记录时，会从第一个睡眠阶段开始前的 `crop_wake_mins` 分钟和最后一个睡眠阶段结束后的 `crop_wake_mins` 分钟开始截断数据，以关注主要的睡眠部分。默认为 `30` 分钟。
- **root_path** (`str`): Sleep-EDF "sleep-cassette" 数据集的根目录路径。该目录应包含成对的 `SC*.edf` (PSG信号) 和 `SC*Hypnogram.edf` (睡眠分期标注) 文件。默认为 `'./sleep-edf-database-expanded-1.0.0/sleep-cassette'`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他 `BaseDataset` 参数如 `start_offset`, `end_offset`, `include_end`（这些通常在此数据集中保持默认值0，因为分段主要由MNE的30秒分期事件驱动），以及各种变换回调函数和IO参数，功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:

  - 每个记录包含一个 `*PSG.edf` 文件（包含多通道生理信号）和一个对应的 `*Hypnogram.edf` 文件（包含睡眠阶段标注）。`set_records` 方法（使用 `list_records` 辅助函数）负责查找并配对这些文件。

- 支持的信号类型

   (从 *PSG.edf文件加载):

  - `eeg`: 脑电信号 (2通道: 'Fpz-Cz', 'Pz-Oz')。
  - `eog`: 眼电信号 (1通道: 'horizontal')。
  - `rsp`: 呼吸信号 (1通道: 'oro-nasal' - 口鼻呼吸气流)。
  - `emg`: 肌电信号 (1通道: 'submental' - 颏下肌)。
  - `temp`: 体温信号 (1通道: 'rectal' - 直肠温度)。

- 采样率:

  - 从每个记录的 `.edf` 文件头信息中动态读取 (`raw.info['sfreq']`)。对于 "sleep-cassette" 数据，EEG、EOG、EMG 通常为 **100 Hz**，而呼吸和体温信号通常为 **1 Hz**。

- 标签信息:

  - `'stage'`: 睡眠阶段标签，从 `*Hypnogram.edf` 文件中的注释提取。
  - 睡眠阶段通常以30秒为单位进行分期。
  - 标签映射关系 (0索引):
    - `0`: 'W' (Wake, 清醒)
    - `1`: '1' (NREM Stage 1, N1期)
    - `2`: '2' (NREM Stage 2, N2期)
    - `3`: '3' (NREM Stage 3, N3期)
    - `4`: '4' (NREM Stage 4, N4期) (在一些评分标准中，N3和N4合并为N3)
    - `5`: 'R' (REM, 快速眼动期)
    - 未定义或运动伪迹期可能被映射为 `-1` 或被忽略。

- **分段 (`segment_split` 的特定行为)**: 此类重写了 `segment_split` 方法。它首先根据 `labels['segments']` (由 `read_record` 生成，每个元素代表一个30秒的睡眠分期) 来提取每个信号类型的有效数据段。然后，它将**所有有效的30秒数据段**按信号类型分别堆叠起来。因此，除非应用了如 `SlideWindow` 等进一步的变换，否则 `BaseDataset` 的 `process_record` 在处理每个原始记录文件时，主要针对的是这个聚合后的“整段”数据（其中第一维是分期数量）。

**使用示例**

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

[`返回目录`](#tyeedataset)

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

`SleepEDFTelemetryDataset` 类用于处理 Sleep-EDF Database Expanded 中的 "sleep-telemetry" (ST) 部分。该子数据集包含使用遥测设备记录的多导睡眠图，主要用于研究年龄对睡眠的影响。此类与 `SleepEDFCassetteDataset` 非常相似，但处理的信号通道略有不同。

**主要初始化参数**

- **crop_wake_mins** (`int`): 同 `SleepEDFCassetteDataset`，用于截断记录两端的清醒期。默认为 `30` 分钟。
- **root_path** (`str`): Sleep-EDF "sleep-telemetry" 数据集的根目录路径。该目录应包含成对的 `ST*.edf` (PSG信号) 和 `ST*Hypnogram.edf` (睡眠分期标注) 文件。默认为 `'./sleep-edf-database-expanded-1.0.0/sleep-telemetry'`。
- 其他 `BaseDataset` 参数如 `start_offset`, `end_offset`, `include_end`（这些通常在此数据集中保持默认值0，因为分段主要由MNE的30秒分期事件驱动），以及各种变换回调函数和IO参数，功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 与 "sleep-cassette" 类似，每个记录包含一个 `*PSG.edf` 文件和一个 `*Hypnogram.edf` 文件。
- 支持的信号类型(从 *PSG.edf文件加载):
  - `eeg`: 脑电信号 (2通道: 'Fpz-Cz', 'Pz-Oz')。
  - `eog`: 眼电信号 (1通道: 'horizontal')。
  - `emg`: 肌电信号 (1通道: 'submental' - 颏下肌)。
  - **注意**: 与 "sleep-cassette" 不同，遥测数据通常不包含呼吸和体温信号。
- 采样率:
  - 从 `.edf` 文件头信息中读取。EEG, EOG, EMG 通常为 **100 Hz**。
- 标签信息:
  - `'stage'`: 睡眠阶段标签，从 `*Hypnogram.edf` 文件中的注释提取。
  - 睡眠阶段通常以30秒为单位进行分期。
  - 标签映射关系 (0索引):
    - `0`: 'W' (Wake, 清醒)
    - `1`: '1' (NREM Stage 1, N1期)
    - `2`: '2' (NREM Stage 2, N2期)
    - `3`: '3' (NREM Stage 3, N3期)
    - `4`: '4' (NREM Stage 4, N4期) (在一些评分标准中，N3和N4合并为N3)
    - `5`: 'R' (REM, 快速眼动期)
    - 未定义或运动伪迹期可能被映射为 `-1` 或被忽略。
- **分段 (`segment_split` 的特定行为)**: 此类重写了 `segment_split` 方法。它首先根据 `labels['segments']` (由 `read_record` 生成，每个元素代表一个30秒的睡眠分期) 来提取每个信号类型的有效数据段。然后，它将**所有有效的30秒数据段**按信号类型分别堆叠起来。因此，除非应用了如 `SlideWindow` 等进一步的变换，否则 `BaseDataset` 的 `process_record` 在处理每个原始记录文件时，主要针对的是这个聚合后的“整段”数据（其中第一维是分期数量）。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`TUABDataset` 类用于处理 TUAB (Temple University Hospital Abnormal EEG Corpus) 数据集。该数据集包含大量的临床脑电图（EEG）记录，每个记录被标注为“正常”或“异常”。此类继承自 `BaseDataset`，并实现了从 `.edf` 文件加载EEG信号及从目录结构中提取对应标签的逻辑。特别地，它通常处理使用特定预处理（如 `01_tcp_ar` 平均参考蒙太奇）的EDF文件。

**主要初始化参数**

- **root_path** (`str`): TUAB 数据集的根目录路径，例如指向包含 `train` 或 `eval` 等子目录的路径。此类会递归查找路径中包含 `'01_tcp_ar'` 的子目录下的 `.edf` 文件。默认为 `'./tuh_eeg_abnormal/v3.0.1/edf/train'` (请根据实际路径修改)。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他 `BaseDataset` 参数如 `start_offset`, `end_offset`, `include_end`（这些在此数据集中通常保持默认值0，因为整个记录被视为一个初始段，后续通常通过变换如 `SlideWindow` 进行分段），以及各种变换回调函数和IO参数，功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 信号数据存储在 `.edf` 文件中。`set_records` 方法会递归地在 `root_path` 下查找那些其路径中包含 `'01_tcp_ar'` 字符串的目录下的 `.edf` 文件。
- 支持的信号类型:
  - `eeg`: 脑电信号。
- 采样率:
  - **256Hz**，从每个 `.edf` 文件的头信息中动态读取 (`Rawdata.info['sfreq']`)。
- 通道信息:
  - 从 `.edf` 文件中读取原始通道名称。
  - 通道名称会被标准化（例如，从 `EEG FP1-REF` 简化为 `FP1`）。
- 标签信息:
  - `'label'`: 二分类标签，表示EEG记录是“正常”(normal) 还是“异常”(abnormal)。

**使用样例**

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

[`返回目录`](#tyeedataset)

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

`TUEVDataset` 类用于处理 TUH EEG Events (TUEV) Corpus 数据集。该数据集包含大量的临床脑电图（EEG）记录，并标注了多种类型的EEG事件。此类继承自 `BaseDataset`，并实现了从 `.edf` 文件加载EEG信号及从对应的 `.rec`（事件记录）文件加载事件标注的逻辑。

**一个重要的数据处理特性**：此类在 `process_record` 方法中会将原始EEG信号数据在时间轴上**复制并拼接三次** (`data = np.concatenate([data, data, data], axis=1)`)。之后，从 `.rec` 文件读取的事件时间点会加上一个等于原始信号长度（单次）的偏移量，这意味着事件时间点实际上是相对于这个三倍长度信号的中间部分来定位的。`BaseDataset` 的 `segment_split` 方法随后会使用 `start_offset` 和 `end_offset` 参数（相对于这些调整后的事件时间点）来提取数据段。

**主要初始化参数**

- **root_path** (`str`): TUH EEG Events 数据集的根目录路径，该目录应递归包含 `.edf` 文件及其对应的 `.rec` 事件文件。默认为 `'./tuh_eeg_events/v2.0.1/edf/train'` (请根据实际路径修改)。
- **start_offset** (`float`): 相对于每个（调整后的）事件标记的开始时间（秒），用于分段提取。默认为 `-2.0` 秒 (即事件前2秒)。
- **end_offset** (`float`): 相对于每个（调整后的）事件标记的结束时间（秒），用于分段提取。默认为 `2.0` 秒 (即事件后2秒)。
- **include_end** (`bool`): 分段时是否包含 `end_offset` 计算后的结束时间点对应的样本。默认为 `False`。
- **io_path** (`Union[None, str]`): 用于缓存预处理数据的路径。
- 其他 `BaseDataset` 参数如 `before_segment_transform`, `offline_signal_transform`, `offline_label_transform`, `online_signal_transform`, `online_label_transform`, `io_size`, `io_chunks`, `io_mode`, `num_worker`, `lazy_threshold`, `verbose` 功能同 `BaseDataset`。

**数据集特性**

- 原始数据文件:
  - 信号数据存储在 `.edf` 文件中。
  - 事件标注（类型和时间）存储在与 `.edf` 文件同名但扩展名为 `.rec` 的文本文件中。`set_records` 方法会递归查找 `root_path` 下的所有 `.edf` 文件。
- 支持的信号类型:
  - `eeg`: 脑电信号。
- 采样率:
  - **256Hz**，从每个 `.edf` 文件的头信息中动态读取 (`Rawdata.info['sfreq']`)。
- 通道信息:
  - 从 `.edf` 文件中读取原始通道名称，通道名称会被标准化（例如，从 `EEG FP1-REF` 简化为 `FP1`）。
- 标签信息:
  - `'event'`: EEG事件类型标签，从 `.rec` 文件中读取。

**使用样例**

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
[`返回目录`](#tyeedataset)
