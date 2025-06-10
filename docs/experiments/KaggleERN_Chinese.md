# Kaggle ERN - 事件相关负波检测

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | KaggleERN                                                    |
| **信号类型** | EEG                                                          |
| **分析任务** | 事件相关负波检测 (ERN Detection)，一个二分类任务，旨在判断受试者是否对刺激产生了错误相关的负波反应。 |
| **使用模型** | EEGPT                                                        |
| **参考论文** | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| **原始代码** | https://github.com/BINE022/EEGPT                             |

本实验旨在使用 Tyee 框架，在 **Kaggle ERN** 数据集上，通过对 **EEGPT** 预训练模型进行**线性探查 (Linear Probing)** 来完成 **ERN检测** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中应用大型预训练模型进行线性评估的实用范例。

------

## 2. 准备工作

### 2.1 数据下载

- **下载地址**：[KaggleERN](https://www.kaggle.com/c/inria-bci-challenge/data)

### 2.2 目录结构

请将下载并解压的数据集按以下结构存放。**注意**：原始下载的 `TrainLabels.csv` 和 `true_labels.csv` 文件通常与 `train/` 和 `test/` 文件夹同级，需要手动移动到对应的文件夹内：

```
/path/to/data/KaggleERN/
├── train/
│   ├── Data_S02_Sess01.csv
│   ├── ...
│   └── TrainLabels.csv          # 需要移动到train文件夹内
└── test/
    ├── Data_S01_Sess01.csv
    ├── ...
    └── true_labels.csv          # 需要移动到test文件夹内
```

**重要说明**：
- 原始下载后，标签文件 `TrainLabels.csv` 和 `true_labels.csv` 可能位于数据集根目录
- 请确保将 `TrainLabels.csv` 移动到 `train/` 文件夹内
- 请确保将 `true_labels.csv` 移动到 `test/` 文件夹内
- 这样的结构便于 Tyee 框架正确读取数据和标签

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **EEGPT** 模型。本实验采用**线性探查 (Linear Probing)** 策略，即在训练过程中**冻结 EEGPT 主干网络的所有参数**，只训练新添加的几个线性层。

### 3.2 预训练权重说明

在本实验中，我们加载了 **EEGPT** 的**官方上游预训练权重** (`eegpt_mcae_58chs_4s_large4E.ckpt`) 作为模型的起点。

- **权重来源**：该预训练权重由模型原作者发布。

- **下载链接**：https://figshare.com/s/e37df4f8a907a866df4b
- **使用方法**：请将下载的权重文件放置于配置文件中 `model.load_path` 指定的路径下，模型在初始化时会自动加载此权重。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`KFoldCross`)**: 本实验采用 **4折交叉验证 (4-Fold Cross-Validation)**。数据在划分前会按受试者ID (`group_by: subject_id`) 进行分组，以确保同一受试者的所有数据都属于同一折，避免数据泄露。

### 4.2 数据处理流程

本实验的数据预处理流程完全定义在配置文件的 `offline_signal_transform` 部分。其核心步骤包括：

1. **归一化与缩放 (`MinMaxNormalize`, `Offset`, `Scale`)**: 对信号进行 Min-Max 归一化后，通过偏移和缩放将数值范围调整到 `[-1, 1]`。
2. **通道选择 (`PickChannels`)**: 挑选出标准的 19 个 EEG 通道。

### 4.3 任务定义

- **任务类型**: `kaggleern_task.KaggleERNTask`
- 核心逻辑:
  - **优化器参数设置 (`set_optimizer_params`)**: 该方法被定制为**仅返回新添加的 `chan_conv`, `linear_probe1`, `linear_probe2` 层的参数**进行优化，从而实现了对 EEGPT 主干网络的冻结。
  - **训练/验证步骤**: 接收 EEG 信号 `x` 和标签 `label`，通过模型得到预测值 `pred`，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失。

### 4.4 训练策略

- **优化器**: 使用 **AdamW**，权重衰减为 `0.01`。
- **学习率调度器 (`OneCycleScheduler`)**: 采用 **One-Cycle LR** 策略，在 100 个 epoch 内，学习率先从低处增长到最大学习率 `4e-4`（在前20%的周期内），然后再余弦退火下降。
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
  dataset: kaggleern_dataset.KaggleERNDataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/KaggleERN/train'
    test: '/mnt/ssd/lingyus/KaggleERN/test'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_kaggleern/train"
    test: "/mnt/ssd/lingyus/tyee_kaggleern/test"
  io_mode: hdf5
  io_chunks: 400
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_kaggleern/split
      group_by: subject_id
      n_splits: 4
      shuffle: false
  
  offline_signal_transform:
    - select: MinMaxNormalize
      source: eeg
      target: eeg
    - select: Offset
      offset: -0.5
      source: eeg
      target: eeg
    - select: Scale
      scale_factor: 2.0
      source: eeg
      target: eeg
    - select: PickChannels
      channels: ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
      source: eeg
      target: eeg

lr_scheduler:
  select: OneCycleScheduler
  max_lr: 4e-4
  epochs: 100
  pct_start: 0.2

model:
  select: eegpt.linear_probe_EEGPT_KaggleERN.LitEEGPTCausal
  load_path: /home/lingyus/code/PRL/models/eegpt/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt

optimizer:
  lr: 4e-4
  select: AdamW
  weight_decay: 0.01

task:
  loss:
    select: CrossEntropyLoss
  select: kaggleern_task.KaggleERNTask

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
   python main.py --config config/kaggleern.yaml
   ```

   *(请将 `config/kaggleern.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。由于本实验采用4折交叉验证，下表展示的是**所有折上测试结果的平均性能**。

|      | Balanced Accuracy | Cohen Kappa | ROC AUC |
| ---- | ----------------- | ----------- | ------- |
| Tyee | 61.17             | 22.68       | 67.15   |
| 官方 | 58.37             | 18.82       | 66.21   |

