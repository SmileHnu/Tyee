# Sleep-EDF - 睡眠分期

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | Sleep-EDF Expanded (Sleep-EDF-39), SC-subjects 子集          |
| **信号类型** | EEG (Fpz-Cz), EOG (horizontal)                               |
| **分析任务** | 睡眠分期 (Sleep Staging)，5分类任务 (W, N1, N2, N3, R)       |
| **使用模型** | SalientSleepNet                                              |
| **参考论文** | [SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://arxiv.org/abs/2105.13864) |
| **原始代码** | https://github.com/ziyujia/SalientSleepNet                   |

本实验旨在使用 Tyee 框架，在 **Sleep-EDF-39** 数据集上，通过 **SalientSleepNet** 模型完成 **基于EEG和EOG信号的睡眠分期** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中进行多模态时序数据联合分析的实用范例。

------

## 2. 准备工作

- **下载地址**：[Sleep-EDF Database on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

- **数据集子集说明**：本实验使用的是 Sleep-EDF Expanded 数据集中的 

  Sleep Cassette (SC) 录音部分的前20位受试者，因为包含20个受试者被加叫做Sleep-EDF-20,也因为包含39个`.edf`文件被叫做Sleep-EDF-39。您需要从下载的 `sleep-cassette`文件夹中，提取以下 20位受试者的 `.edf` 文件（包括PSG和Hypnogram）：

  - `SC4001`, `SC4011`, `SC4021`, `SC4031`, `SC4041`, `SC4051`, `SC4061`, `SC4071`, `SC4081`, `SC4091`, `SC4101`, `SC4111`, `SC4121`, `SC4131`, `SC4141`, `SC4151`, `SC4161`, `SC4171`, `SC4181`, `SC4191`

- **目录结构**：请将提取出的文件按以下结构存放：

  ```
  /path/to/data/sleep-edf-39/
  ├── SC4001E0-PSG.edf
  ├── SC4001EC-Hypnogram.edf
  ├── SC4011E0-PSG.edf
  └── ...
  ```

------

## 3. 模型配置

本实验选用 **SalientSleepNet** (`TwoStreamSalientModel`) 模型。其关键架构参数在配置文件的 `model.config` 部分设置如下 ：

- `sleep_epoch_len`: 3000 (每个睡眠时期的长度, 30s * 100Hz) 
- `sequence_epochs`: 20 (每个输入样本包含的连续睡眠时期数量) 
- `filters`: [16, 32, 64, 128, 256] (U-Net编码器各阶段的滤波器数量) 
- `kernel_size`: 5 (卷积核大小) 
- `pooling_sizes`: [10, 8, 6, 4] (U-Net编码器各阶段的池化大小) 
- `u_depths`: [4, 4, 4, 4] (U-Net单元的深度) 

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`KFoldCross`)**: 对数据集中的20位受试者采用 20折交叉验证 ，这等同于留一交叉验证 (Leave-One-Subject-Out) 。在每一折中，一位受试者的数据作为验证集，其余19位受试者的数据作为训练集 。

### 4.2 数据处理流程

本实验的数据预处理流程在配置文件中定义，并参考了公开仓库 **SleepDG** 的实现。

- **预处理参考**: [SleepDG (GitHub)](https://github.com/wjq-learning/SleepDG)
- **核心步骤**:
  1. **通道选择 (`PickChannels`)**: 仅选择 'Fpz-Cz' 导联的EEG信号 。
  2. **标签映射 (`Mapping`)**: 将原始的睡眠阶段（包括N4）映射为标准的5个类别（W, N1, N2, N3, R) 。
  3. **上下文滑窗 (`SlideWindow`)**: 将每个30秒的睡眠时期（epoch）与其前后各9个时期组合，形成一个包含 20个epoch的长序列作为模型的单个输入样本 。
- **关于样本数量的特别说明**: 由于预处理流程与原论文可能存在细微差别，本实验处理后的样本标签数量与原论文有轻微差异，具体如下：
  - **原论文标签数**: {W: 8285, N1: 2804, N2: 17799, N3: 5703, REM: 7717}
  - **本实验标签数**: {W: 7619, N1: 2804, N2: 17799, N3: 5703, REM: 7715}

### 4.3 任务定义

- **任务类型**: `sleepedfx_task.SleepEDFxTask`
- **核心逻辑**:
  - **损失函数**: 采用带权重的交叉熵损失 (`CrossEntropyLoss`)，权重 `[1.0, 1.80, 1.0, 1.20, 1.25]`用于缓解睡眠分期任务中普遍存在的类别不均衡问题 。
  - **训练/验证步骤**: 接收EEG和EOG长序列作为双路输入，通过模型得到预测值 。由于模型一次预测20个epoch，需要将标签和预测值展平 (`reshape`) 后再计算损失 。

### 4.4 训练策略

- **优化器**: 使用 Adam，学习率为 0.001 。
- **训练周期**: 共训练 60个 epoch 。
- **评估指标**: 使用准确率 (`accuracy`) 作为核心评估指标来选择最佳模型 。

### 4.5 完整配置文件

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 12
  dataset: sleepedfx_dataset.SleepEDFCassetteDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/sleep-edf-20'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_sleepedfx_20/train"
  io_mode: hdf5
  io_chunks: 20
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_sleepedfx_20/split_17_20
      group_by: subject_id
      n_splits: 20
      shuffle: false

  before_segment_transform:
    - select: PickChannels
      channels: ['Fpz-Cz']
      source: eeg
      target: eeg
  
  offline_signal_transform:
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: eeg
      target: eeg
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: eog
      target: eog
    - select: Select
      key: ['eeg', 'eog']
  
  offline_label_transform:
    - select: Mapping
      mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4} # N4 -> N3
      source: stage
      target: stage
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: stage
      target: stage
  
  online_signal_transform:
    - select: Compose
      transforms:
        - select: Transpose
          axes: [1, 0, 2]
        - select: Reshape
          shape: [1, -1]
        - select: ExpandDims
          axis: -1
      source: eeg
      target: eeg
    - select: Compose
      transforms:
        - select: Transpose
          axes: [1, 0, 2]
        - select: Reshape
          shape: [1, -1]
        - select: ExpandDims
          axis: -1
      source: eog
      target: eog

model:
  select: salient_sleep_net.TwoStreamSalientModel
  config:
    sleep_epoch_len: 3000
    preprocess:
      sequence_epochs: 20
    train:
      filters: [16, 32, 64, 128, 256]
      kernel_size: 5
      pooling_sizes: [10, 8, 6, 4]
      dilation_sizes: [1, 2, 3, 4]
      activation: 'relu'
      u_depths: [4, 4, 4, 4]
      u_inner_filter: 16
      mse_filters: [8, 16, 32, 64, 128]
      padding: 'same'

optimizer:
  lr:  0.001
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
    weight: [1.0, 1.80, 1.0, 1.20, 1.25]
  select: sleepedfx_task.SleepEDFxTask

trainer:
  fp16: false
  total_epochs: 60
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, precision_macro, f1_macro, recall_macro]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```yaml
   python main.py --config config/sleep_edf.yaml
   ```

   *(请将 `config/sleep_edf.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的验证性能。由于本实验采用20折交叉验证，下表展示的是**所有折上验证集结果的平均性能**。

|      | 准确率 (Accuracy) | F1 (Macro) |
| ---- | ----------------- | ---------- |
| Tyee | 88.25             | 82.77      |
| 官方 | 87.5              | 83         |