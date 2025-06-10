# BCICIV 2a - 运动想象

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | BCI Competition IV Dataset 2a (BCICIV 2a)                    |
| **信号类型** | EEG                                                          |
| **分析任务** | 运动想象分类 (Motor Imagery Classification)，4分类任务       |
| **使用模型** | Conformer                                                    |
| **原始论文** | [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/abstract/document/9991178/) |
| **原始代码** | https://github.com/eeyhsong/EEG-Conformer                    |
| **特别说明** | 本实验为单受试者独立实验，下文以受试者A09为例进行说明。      |

本实验旨在使用 Tyee 框架，在 **BCICIV 2a** 数据集上，通过 **Conformer** 模型完成 **运动想象四分类** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为使用 Tyee 进行运动想象分析的实用范例。

------

## 2. 准备工作 

### 2.1 数据下载

BCICIV 2a 数据集的信号数据和标签是分开提供的，需要分别下载：

- **信号数据下载**：[BCICIV_2a_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip)
  - 包含所有受试者的 EEG 信号数据（.gdf 格式）
  - 包含训练集和测试集的信号，但 .gdf 文件中的测试集**没有标签**

- **完整标签下载**：[true_labels.zip](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)
  - 包含训练集和测试集的完整标签（.mat 格式）
  - 这是官方后续提供的包含测试集真实标签的版本

### 2.2 目录结构

由于本实验是单受试者独立建模，请为每位受试者创建一个单独的文件夹，并将该受试者的训练 (`T`文件) 和评估 (`E`文件) 的 `.gdf` 文件以及对应的 `.mat` 标签文件放入其中。以受试者 `A09`为例，目录结构应如下：

```
/path/to/data/BCICIV_2a/
└── A09/
    ├── A09T.gdf          # 训练集信号数据
    ├── A09T.mat          # 训练集标签
    ├── A09E.gdf          # 测试集信号数据
    └── A09E.mat          # 测试集标签
```

**重要说明**：
- `.gdf` 文件来自 `BCICIV_2a_gdf.zip`
- `.mat` 文件来自 `true_labels.zip`
- 确保信号文件和标签文件的受试者编号一致

------

## 3. 模型配置

本实验选用 **Conformer** 模型进行运动想象分类，其关键参数在配置文件中设置如下：

- **`n_outputs`**: 4 (对应4个运动想象类别)
- **`n_chans`**: 22 (对应22个EEG通道)
- **`n_times`**: 1000 (每个样本的时间点数)

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。我们选择 **Adam** 作为优化器，初始学习率为 **0.0002**，并采用标准的**交叉熵损失 (CrossEntropyLoss)**。训练总共进行 **2000 个 epoch**。

### 4.1数据集划分

本实验严格遵循 BCICIV 2a 数据集官方的划分方式：

- **划分策略**: 使用官方提供的训练集（`T` 文件，对应 `session_id=0`）进行模型训练，并使用评估集（`E` 文件，对应 `session_id=1`）作为验证集来评估模型性能。
- **配置实现**: 在配置文件中，我们通过 `HoldOutCross` 策略，并设置 `group_by: session_id` 和 `val_size: 0.5` 来实现这一划分。

### 4.2 数据处理流程

我们对数据应用了**离线 (Offline)** 和 **在线 (Online)** 两阶段处理：

1. 离线处理:
   - **带通滤波 (`Cheby2Filter`)**: 对EEG信号应用 4-40 Hz 的带通滤波。
2. 在线处理:
   - **Z-Score 标准化 (`ZScoreNormalize`)**: 使用该受试者**训练集**的全局均值和标准差进行标准化。这组 `mean` 和 `std` 值是针对每个受试者提前计算好的，具体计算方法请参见**附录**。

### 4.3 任务定义

- **任务类型**: `bciciv2a_task.BCICIV2aTask`
- **核心逻辑**: 该任务负责接收模型的预测和数据标签，并通过**交叉熵损失**计算损失值，以驱动模型训练。

### 4.4 完整配置文件

以下是本实验 (以A09受试者为例) 使用的完整配置：

```yaml
common:
  seed: 2021
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 72
  dataset: bciciv2a_dataset.BCICIV2ADataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/BCICIV_2a/A09'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_bciciv2a/A09"
  io_mode: hdf5
  split: 
    select: HoldOutCross
    init_params:
      split_path: '/mnt/ssd/lingyus/tyee_bciciv2a/split/A09'
      group_by: session_id
      val_size: 0.5
      random_state: 4523

  offline_signal_transform:
    - select: Cheby2Filter
      l_freq: 4
      h_freq: 40
      source: eeg
      target: eeg
    - select: Select
      key: ['eeg']

  online_signal_transform:
    - select: ZScoreNormalize
      mean: -0.000831605672567208
      std: 9.915488018511994
      source: eeg
      target: eeg

model:
  select: conformer.Conformer
  n_outputs: 4
  n_chans: 22
  n_times: 1000

optimizer:
  lr: 0.0002
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: bciciv2a_task.BCICIV2aTask

trainer:
  fp16: true
  total_epochs: 2000
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, cohen_kappa]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```yaml
   python main.py --config config/bciciv2a.yaml
   ```

   *(请将 `main.py` 和配置文件路径替换为你的实际执行脚本和路径)*

3. **查看结果**：实验的所有输出，包括**训练日志**、**模型权重 (.pt 文件)** 和 **TensorBoard 结果**，都将保存在配置文件中 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹，例如：`./experiments/2025-06-07/10-30-00/`。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的验证结果。下表展示的是整个训练过程中，模型在验证集上所取得的**最佳性能**。

| 受试者        | 准确率 (Accuracy) |
| ------------- | ----------------- |
| 1             | 85.07             |
| 2             | 59.03             |
| 3             | 91.67             |
| 4             | 77.08             |
| 5             | 48.26             |
| 6             | 59.03             |
| 7             | 89.58             |
| 8             | 82.64             |
| 9             | 85.07             |
| **Tyee 平均** | **75.27**         |
| **官方结果**  | **74.91**         |

注：官方实验中使用了数据增强，但该实验未采用数据增强，所对比的实验结果也是未使用数据增强的

## 附录：训练集全局均值和标准差计算

为了保证验证集和测试集在标准化过程中不泄露任何信息，我们必须使用且仅使用**训练集**的数据来计算全局的均值（`mean`）和标准差（`std`）。这些计算出的值随后被硬编码到配置文件的 `online_signal_transform` 部分，以应用于所有数据。

以下是为单个受试者（以A09为例）计算这些统计值的示例脚本：

```python
import numpy as np
from tyee.dataset import BCICIV2ADataset # 假设这是你的数据集类，请替换为实际导入路径
from tyee.dataset.transform import Cheby2Filter, Select # 假设的离线转换类

# 1. 定义与配置文件一致的离线转换
offline_signal_transform = [
    Cheby2Filter(l_freq=4, h_freq=40, source='eeg', target='eeg'),
    Select(key=['eeg'])
]

# 2. 初始化数据集实例，Tyee会自动进行离线处理和缓存
# 注意：此时不应用在线的ZScoreNormalize
dataset = BCICIV2ADataset(
    root_path='/mnt/ssd/lingyus/BCICIV_2a/A09',
    io_path='/mnt/ssd/lingyus/tyee_bciciv2a/A09',
    io_mode='hdf5',
    io_chunks=750,
    offline_signal_transform=offline_signal_transform
)

# 3. 根据session_id筛选出训练集样本的索引
# 在BCICIV 2a数据集中，'T'文件通常对应session 0, 'E'文件对应session 1
train_indices = dataset.info[dataset.info['session_id'] == 0].index.tolist()

# 4. 收集所有训练集样本的EEG数据
all_train_eeg = []
for idx in train_indices:
    eeg_data = dataset[idx]['eeg']  # shape: (channels, time_points)
    all_train_eeg.append(eeg_data)

# 5. 拼接所有数据并计算全局均值和标准差
# 沿着时间轴拼接，以计算所有通道和时间点的总体统计量
full_train_data = np.concatenate(all_train_eeg, axis=-1) 
mean = np.mean(full_train_data)
std = np.std(full_train_data)

print(f"Statistics for Subject A09:")
print(f"Mean: {mean}")
print(f"Std: {std}")
```

你需要为每个受试者分别运行此脚本，并将得到的 `mean` 和 `std` 填入各自的配置文件中。