# BCICIV 4 - 手指运动解码

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | BCI Competition IV Dataset 4                                 |
| **信号类型** | ECoG (皮层脑电图)                                            |
| **分析任务** | 手指运动解码 (Finger Movement Decoding)，一个回归任务，旨在预测五个手指的屈曲程度。 |
| **使用模型** | FingerFlex (AutoEncoder1D)                                   |
| **参考论文** | [FingerFlex: Inferring Finger Trajectories from ECoG signals](https://arxiv.org/abs/2211.01960) |
| **原始代码** | https://github.com/Irautak/FingerFlex                        |
| **特别说明** | 本实验为单受试者独立实验。                                   |

本实验旨在使用 Tyee 框架，在 **BCICIV 4** 数据集上，通过 **FingerFlex** 模型完成 **ECoG信号到手指运动轨迹的解码** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中进行脑机接口回归任务的实用范例。

------

## 2. 准备工作

### 2.1 数据下载

BCICIV 4 数据集的信号数据和测试集标签是分开提供的，需要分别下载：

- **信号数据下载**：[BCICIV_4_mat.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip)
  - 包含所有受试者的 ECoG 信号数据和训练集标签（.mat 格式）
  - 文件命名格式：`sub{X}_comp.mat`（如 `sub1_comp.mat`, `sub3_comp.mat`）
  - 包含训练集的完整数据，但**没有测试集标签**

- **测试集标签下载**：[true_labels.zip](https://www.bbci.de/competition/iv/results/ds4/true_labels.zip)
  - 包含测试集的真实标签（.mat 格式）
  - 文件命名格式：`sub{X}_testlabels.mat`（如 `sub1_testlabels.mat`, `sub3_testlabels.mat`）
  - 这是官方后续提供的包含测试集真实标签的版本

### 2.2 目录结构

由于本实验是单受试者独立建模，请为每位受试者创建一个单独的文件夹，并将该受试者的信号数据文件和测试标签文件放入其中。以受试者1为例，目录结构应如下：

```
/path/to/data/BCICIV4/
└── sub1/
    ├── sub1_comp.mat          # 训练集信号数据和标签
    └── sub1_testlabels.mat    # 测试集标签
```

**重要说明**：
- `sub{X}_comp.mat` 文件来自 `BCICIV_4_mat.zip`
- `sub{X}_testlabels.mat` 文件来自 `true_labels.zip`
- 确保信号文件和标签文件的受试者编号一致
- 原始的 `sub{X}_comp.mat` 文件包含训练数据和训练标签，但缺少测试集标签

------

## 3. 模型配置

本实验选用 **FingerFlex (`AutoEncoder1D`)** 模型进行手指运动解码。该模型在本实验中**从零开始 (from scratch)** 进行训练。其关键架构参数在配置文件中设置如下：

- **`channels`**: `[32, 32, 64, 64, 128, 128]` (编码器中各卷积层的输出通道数)
- **`kernel_sizes`**: `[7, 7, 5, 5, 5]` (编码器中各卷积层的核大小)
- **`strides`**: `[2, 2, 2, 2, 2]` (编码器中各卷积层的步长)
- **`n_electrodes`**: 62 (输入的ECoG电极数量)
- **`n_freqs`**: 40 (每个电极的频谱特征数量)
- **`n_channels_out`**: 5 (最终输出的通道数，对应五个手指)

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略**: 本实验遵循官方的数据集划分。由于官方训练集和测试集（此处用作验证集）包含在同一个文件中，我们将其视为两个独立的试验 (`trial`)。在配置文件中，我们通过 `HoldOutCross` 策略，并设置 `group_by: trial_id` 和 `val_size: 0.5`，从而精确地将一个 `trial` 用于训练，另一个用于验证。

### 4.2 数据处理流程

**重要说明**：由于本实验的数据预处理流程较为复杂，它是在 `dataset` 类（`bciciv4_dataset.py`）中直接通过代码实现的，**并未在 `config.yaml` 文件中定义**。详细的变换代码请参见**附录 A.1**。

其主要步骤概括如下：

1. **信号 (ECoG) 处理**: Z-score标准化、共平均参考 (CAR)、带通滤波、陷波滤波、小波变换频谱提取、降采样、鲁棒标准化、滑窗等。
2. **标签 (Fingers) 处理**: 降采样、插值、Min-Max归一化、滑窗等。

### 4.3 任务定义

- **任务类型**: `bciciv4_task.BCICIV4Task`
- 核心逻辑: 
  - **损失函数**: 采用了一个复合损失函数，即 **0.5 \* MSE_Loss + 0.5 \* (1 - Correlation)**。这个损失函数同时优化了预测轨迹与真实轨迹的均方误差以及它们之间的相关性。
  - **训练/验证步骤**: 接收 ECoG 信号 `x` 和手指数据 `label`，通过模型得到预测值 `pred`，然后使用上述的复合损失函数进行计算。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **8.42e-5**，并设置了 `1e-6` 的权重衰减。
- **训练周期**: 共训练 **20个 epoch**。
- **评估指标**: 使用**平均相关系数 (`mean_cc`)** 作为核心评估指标，并在训练过程中选择 `mean_cc` 最高的模型作为最佳模型。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: bciciv4_dataset.BCICIV4Dataset
  num_workers: 8
  root_path:
    train: '/home/lingyus/data/BCICIV4/sub1'
  io_path:
    train: "/home/lingyus/data/BCICIV4/sub1/processed_test"
  io_mode: hdf5
  split:
    select: HoldOutCross
    init_params:
      split_path: /home/lingyus/data/BCICIV4/sub1/split
      group_by: trial_id
      val_size: 0.5
      random_state: 0
      shuffle: true

model:
  select: fingerflex.AutoEncoder1D
  channels: [32, 32, 64, 64, 128, 128]
  kernel_sizes: [7, 7, 5, 5, 5]
  strides: [2, 2, 2, 2, 2]
  dilation: [1, 1, 1, 1, 1]
  n_electrodes: 62
  n_freqs: 40
  n_channels_out: 5

optimizer:
  lr:  8.42e-5
  select: Adam
  weight_decay: 1e-6

task:
  loss:
    select: MSELoss
  select: bciciv4_task.BCICIV4Task

trainer:
  fp16: false
  total_epochs: 20
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: mean_cc
    mode: max
  metrics: [mean_cc]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/bciciv4.yaml
   ```

   *(请将 `config/bciciv4.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹，例如：`./experiments/2025-06-07/12-00-00/`。

------

## 6. 预期结果

成功复现本实验后，你应该能在训练结束时获得与下表相似的最终验证性能。下表展示的是**受试者S1**整个训练过程中，模型在验证集上所取得的**最佳性能**。

|      | 平均相关系数 (Mean Correlation Coefficient) |
| ---- | ------------------------------------------- |
| Tyee | 0.6925                                      |
| 官方 | 0.66                                        |

## 附录

### A.1 数据预处理代码与标准化统计量

本实验的数据预处理流程在 `bciciv4_dataset.py` 中直接定义。

**关于标准化统计量 (`robust_stats` 和 `minmax_stats`) 的说明：** 为了防止验证集信息泄露，我们采用严格的标准化流程：

1. 首先，对数据集执行**除 `RobustNormalize` 和 `MinMaxNormalize` 之外**的所有预处理步骤。
2. 然后，对处理后的数据进行训练集/验证集的划分。
3. **仅在训练集上**计算鲁棒标准化所需的 `center_` (中位数) 和 `scale_` (四分位距)，以及Min-Max归一化所需的 `data_min_` 和 `data_max_`，并将这些统计量保存为文件（如`.npz`）。
4. 最后，在完整的处理流程中，`RobustNormalize` 和 `MinMaxNormalize` 步骤会加载这些从训练集得到的统计量，并将其应用于训练集和验证集，从而保证了两者使用统一的标准化标准。

以下是核心的变换逻辑代码：

```python
# 导入所需的变换类
from tyee.dataset.transform import (
    Compose, NotchFilter, Filter, ZScoreNormalize, RobustNormalize, Reshape,
    CWTSpectrum, Downsample, Crop, Interpolate, MinMaxNormalize, CommonAverageRef,
    Transpose, SlideWindow
)
import numpy as np

# 注意：以下统计量是从训练集预先计算并加载的
# robust_stats = np.load('/path/to/your/robust_scaler_stats0.npz')
# minmax_stats = np.load('/path/to/your/minmax_scaler_stats0.npz')

# --- 信号 (ECoG) 预处理变换 ---
offline_signal_transform = [
    ZScoreNormalize(epsilon=0, axis=1, source='ecog', target='ecog'),
    CommonAverageRef(axis=0, source='ecog', target='ecog'),
    Filter(l_freq=40, h_freq=300, source='ecog', target='ecog'),
    NotchFilter(freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450], source='ecog', target='ecog'),
    CWTSpectrum(freqs=np.logspace(np.log10(40), np.log10(300), 40), output_type='power', n_jobs=6, source='ecg', target='ecg'),
    Downsample(desired_freq=100, source='ecog', target='ecog'),
    Crop(crop_right=20, source='ecog', target='ecog'),
    Transpose(source='ecog', target='ecog'),
    Reshape(shape=(-1, 40*62), source='ecog', target='ecog'),
    RobustNormalize(
        median=robust_stats['center_'], iqr=robust_stats['scale_'], 
        unit_variance=False, quantile_range=(0.1, 0.9), epsilon=0, axis=0, source='ecog', target='ecog'
    ),
    Reshape(shape=(-1, 40, 62), source='ecog', target='ecog'),
    Transpose(source='ecog', target='ecog'),
    SlideWindow(window_size=256, stride=1, source='ecog', target='ecog'),
]

# --- 标签 (Fingers) 预处理变换 ---
offline_label_transform = [
    Downsample(desired_freq=25, source='dg', target='dg'),
    Interpolate(desired_freq=100, kind='cubic', source='dg', target='dg'),
    Crop(crop_left=20, source='dg', target='dg'),
    Transpose(source='dg', target='dg'),
    MinMaxNormalize(
        min=minmax_stats['data_min_'], max=minmax_stats['data_max_'], 
        axis=0, source='dg', target='dg'
    ),
    Transpose(source='dg', target='dg'),
    SlideWindow(window_size=256, stride=1, source='dg', target='dg'),
]
```

### A.2 关于验证集窗口化的特别说明

原 FingerFlex 代码仅对训练集进行滑窗 (`SlideWindow`) 处理，而验证集则作为一个整体进行评估，以获得与真实标签长度一致的连续预测。

为了在 Tyee 框架中实现这一非对称处理，我们采取了如下策略：

1. **统一处理与缓存**：在数据加载和缓存阶段，训练集和验证集**都被** `SlideWindow` 切分为大量重叠的窗口，以保证数据处理流程的一致性。
2. **验证时还原序列**：在进行**验证 (validation)** 时，我们通过特殊的评估逻辑，**忽略**了验证集数据的窗口化结构。框架会直接将完整的、未经切分的验证集 `trial` 视为一个连续的长序列输入模型进行评估。
3. **对齐评估**：这样，模型对整个验证集 `trial` 输出一个等长的连续预测，可以直接与完整的真实标签进行比较，从而计算相关性。

