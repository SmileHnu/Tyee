# PhysioNet P300 - P300电位检测

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | PhysioP300                                                   |
| **信号类型** | EEG                                                          |
| **分析任务** | P300电位检测 (P300 Speller Paradigm)，一个旨在判断EEG信号中是否包含P300事件相关电位的二分类任务。 |
| **使用模型** | EEGPT                                                        |
| **参考论文** | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| **原始代码** | https://github.com/BINE022/EEGPT                             |

本实验旨在使用 Tyee 框架，在 **PhysioNet P300** 数据集上，通过对 **EEGPT** 预训练模型进行**线性探查 (Linear Probing)** 来完成 **P300电位检测** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中应用大型预训练模型进行线性评估的实用范例。

------

## 2. 准备工作

- **下载地址**：[PhysioP300](https://physionet.org/content/erpbci/1.0.0/)

- 目录结构：请将下载并解压的数据集按以下结构存放：

  ```
  /path/to/data/erp-based-brain-computer-interface-recordings-1.0.0/files/
  ├── s01/
  │   └── rc01.edf
  │   └── ...
  ├── s02/
  │   └── rc01.edf
  │   └── ...
  └── ...
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **EEGPT** 模型。我们采用**线性探查 (Linear Probing)** 策略，即在训练过程中**冻结 EEGPT 主干网络的所有参数**，只训练新添加的一个可学习的通道缩放层 (`chan_scale`) 和两个线性层 (`linear_probe1`, `linear_probe2`)。

### 3.2 预训练权重说明

在本实验中，我们加载了 **EEGPT** 的**官方上游预训练权重** (`eegpt_mcae_58chs_4s_large4E.ckpt`) 作为模型的起点。

- **权重来源**：该预训练权重由模型原作者发布。

- **下载链接**：https://figshare.com/s/e37df4f8a907a866df4b
- **使用方法**：请将下载的权重文件放置于配置文件中 `model.load_path` 指定的路径下，模型在初始化时会自动加载此权重。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **特别说明**：原始数据集包含12名受试者，但为了与原论文的实验设置对齐，本实验**排除了 S8, S10, S12 三名受试者**，仅在其余9名受试者的数据上进行。
- **划分策略 (`KFoldCross`)**: 对剩余的9名受试者采用 **9折交叉验证 (9-Fold Cross-Validation)**。数据在划分前会按受试者ID (`group_by: subject_id`) 进行分组，实际上等同于留一交叉验证（Leave-One-Subject-Out）。

### 4.2 数据处理流程

本实验的数据预处理流程完全定义在配置文件的 `offline_signal_transform` 部分。其核心步骤包括：

1. **通道选择 (`PickChannels`)**: 挑选出标准的 64 个 EEG 通道。
2. **基线校正 (`Baseline`)**: 使用每个试验的前 `1435` 个时间点作为基线，从整个试验中减去该基线的平均值。
3. **滤波 (`Filter`)**: 应用 0-120 Hz 的带通滤波。
4. **重采样 (`Resample`)**: 将信号的采样率统一降至 256 Hz。
5. **缩放 (`Scale`)**: 将信号数值乘以 `1e-3`。

### 4.3 任务定义

- **任务类型**: `physiop300_task.PhysioP300Task`
- **核心逻辑**:
  - **优化器参数设置 (`set_optimizer_params`)**: 该方法被定制为**仅返回新添加的 `chan_scale`, `linear_probe1`, `linear_probe2` 层的参数**进行优化，从而实现了对 EEGPT 主干网络的冻结。
  - **训练/验证步骤**: 接收 EEG 信号 `x` 和标签 `label`，通过模型得到预测值 `pred`，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失。

### 4.4 训练策略

- **优化器**: 使用 **AdamW**，权重衰减为 `0.01`。
- **学习率调度器 (`OneCycleScheduler`)**: 采用 **One-Cycle LR** 策略，在 100 个 epoch 内，学习率先从低处线性增长到最大学习率 `4e-4`（在前20%的周期内），然后再余弦退火下降。
- **训练周期**: 共训练 **100个 epoch**。
- **评估指标**: 使用**平衡准确率 (`balanced_accuracy`)** 作为核心评估指标来选择最佳模型，同时计算包括 `roc_auc` 在内的多种详细指标。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 7
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: physiop300_dataset.PhysioP300Dataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/erp-based-brain-computer-interface-recordings-1.0.0'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_physiop300/train"
  io_mode: hdf5
  io_chunks: 512
  include_end: true
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_physiop300/split
      group_by: subject_id
      n_splits: 9
      shuffle: false
  
  offline_signal_transform:
    - select: Compose
      transforms:
        - select: PickChannels
          channels: ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        - select: Baseline
          baseline_end: 1435
          axis: 1
        - select: Filter
          l_freq: 0
          h_freq: 120
          method: iir
        - select: Resample
          desired_freq: 256
          pad: edge
        - select: Scale
          scale_factor: 1e-3
      source: eeg
      target: eeg

lr_scheduler:
  select: OneCycleScheduler
  max_lr: 4e-4
  epochs: 100
  pct_start: 0.2

model:
  select: eegpt.linear_probe_EEGPT_PhysioP300.LitEEGPTCausal
  load_path: /home/lingyus/code/PRL/models/eegpt/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt

optimizer:
  lr: 4e-4
  select: AdamW
  weight_decay: 0.01

task:
  loss:
    select: CrossEntropyLoss
  select: physiop300_task.PhysioP300Task

trainer:
  fp16: true
  total_epochs: 100
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: balanced_accuracy
    mode: max
  metrics: [balanced_accuracy, accuracy, f1_weighted, cohen_kappa, roc_auc]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 和预训练模型路径 (`model.load_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/physiop300.yaml
   ```

   *(请将 `config/physiop300.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。由于本实验采用9折交叉验证，下表展示的是**所有折上验证集最优结果的平均性能**。

|      | Balanced Accuracy | Cohen Kappa | ROC AUC |
| ---- | ----------------- | ----------- | ------- |
| Tyee | 66.51             | 37.74       | 81.16   |
| 官方 | 65.02             | 29.99       | 71.68   |