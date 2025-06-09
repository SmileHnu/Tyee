# TUAB - 异常脑电检测

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | TUH Abnormal EEG Corpus (TUAB) v3.0.1                        |
| **信号类型** | EEG                                                          |
| **分析任务** | 异常脑电检测 (Abnormal EEG Detection)，一个旨在区分正常与异常脑电记录的二分类任务。 |
| **使用模型** | LaBraM (Large Brain Model)                                   |
| **参考论文** | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) |
| **原始代码** | https://github.com/935963004/LaBraM                          |

本实验旨在使用 Tyee 框架，在 **TUAB** 数据集上，通过微调 (fine-tuning) **LaBraM** 预训练模型来完成 **异常脑电信号的二分类** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中应用大型预训练模型进行分类的实用范例。

------

## 2. 准备工作

- **下载地址**：[TUAB](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab)(需要申请和授权访问 TUAB)

- **目录结构**：请将下载并解压的数据集保持原始结构存放：

  ```
  /path/to/data/tuh_eeg_abnormal/v3.0.1/edf/
  ├── train/
  │   ├── abnormal/
  │   └── normal/
  └── eval/
      ├── abnormal/
      └── normal/
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **LaBraM** (`labram_base_patch200_200`) 模型进行异常脑电检测。

### 3.2 模型参数

该模型在本实验中的关键参数在配置文件中设置如下：

- **`nb_classes`**: 2 (对应正常/异常二分类)
- **`drop`**: 0.0 (全连接层中的 Dropout 率)
- **`drop_path`**: 0.1 (Stochastic Depth 的 Drop Path 率)
- **`attn_drop_rate`**: 0.0 (注意力机制中的 Dropout 率)
- **`use_mean_pooling`**: true (使用平均池化来聚合特征)
- **`layer_scale_init_value`**: 0.1 (层缩放（Layer Scale）的初始值)

### 3.3 预训练权重说明

在本实验中，我们加载了 **LaBraM** 的**官方上游预训练权重**作为模型初始化的起点，并在 TUAB 数据集上进行微调。

- **权重来源**：该预训练权重由模型原作者发布。
- **下载链接**：https://github.com/935963004/LaBraM/tree/main/checkpoints
- **使用方法**：请将下载的权重文件放置于配置文件中 `model.finetune` 指定的路径下，Tyee 将在实验开始时自动加载此权重。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`HoldOutCross`)**: 本实验遵循 TUAB 数据集官方的训练集和测试集（此处用作评估集 `eval`）划分。在此基础上，我们将官方的**训练集**进一步划分为新的训练集和验证集。
- **划分比例 (`val_size`)**: 从原始训练集中按受试者ID (`group_by: subject_id`) 进行划分，将 **20%** 的受试者划入验证集，其余 **80%** 作为新的训练集。

### 4.2 数据处理流程

1. 分段前处理 (`before_segment_transform`):
   - **通道选择 (`PickChannels`)**: 挑选出标准的 23 个 EEG 通道。
   - **滤波 (`Filter` & `NotchFilter`)**: 应用 0.1-75 Hz 的带通滤波，并使用 50 Hz 的陷波器去除工频干扰。
   - **重采样 (`Resample`)**: 将信号的采样率统一降至 200 Hz。
2. 离线处理 (`offline_signal_transform`):
   - **滑窗 (`SlideWindow`)**: 将连续信号以 2000 个时间点（10秒）为窗口大小、2000 为步长进行切分，无重叠。
3. 标签处理 (`offline_label_transform`):
   - **标签映射 (`Mapping`)**: 将 'abnormal' 和 'normal' 标签映射为 1 和 0。

### 4.3 任务定义

- **任务类型**: `tuab_task.TUABTask`
- **核心逻辑**: 
  - **模型构建 (`build_model`)**: 负责加载 LaBraM 模型结构，并从指定路径加载官方预训练权重。
  - **优化器参数设置 (`set_optimizer_params`)**: 实现了**层级学习率衰减 (Layer-wise Learning Rate Decay)** 策略，对模型不同深度的层应用不同的学习率。
  - **训练/验证步骤**: 在将数据送入模型前，会进行特定的 `rearrange` 操作以匹配 LaBraM 的输入格式，并对输入信号进行 `/100` 的缩放。

### 4.4 训练策略

- **优化器**: 使用 **AdamW**，基础学习率为 **5e-4**，并设置了 **0.05** 的权重衰减和 **0.65** 的层级学习率衰减。
- **学习率调度器 (`CosineLRScheduler`)**: 采用余弦退火学习率，总周期为50个epoch，并包含前5个epoch的线性预热。
- **损失函数**: 采用带标签平滑（smoothing=0.1）的**交叉熵损失 (`LabelSmoothingCrossEntropy`)**。
- **梯度累积**: `update_interval` 设置为8，表示每累积8个batch的梯度后进行一次模型参数更新。

### 4.5 完整配置文件

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: tuab_dataset.TUABDataset
  num_workers: 18
  root_path:
    train: '/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/train'
    test: '/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/eval'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_tuab/processed_train"
    test: "/mnt/ssd/lingyus/tyee_tuab/processed_eval"
  io_mode: hdf5
  split: 
    select: HoldOutCross
    init_params:
      split_path: /mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/split
      group_by: subject_id
      val_size: 0.2
      random_state: 12345
      shuffle: true

  before_segment_transform:
    - select: Compose
      transforms:
        - select: PickChannels
          channels: ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        - select: Filter
          l_freq: 0.1
          h_freq: 75.0
        - select: NotchFilter
          freqs: [50.0]
        - select: Resample
          desired_freq: 200
      source: eeg
      target: eeg
  
  offline_signal_transform:
    - select: SlideWindow
      window_size: 2000
      stride: 2000
      source: eeg
      target: eeg

  offline_label_transform:
    - select: Mapping
      mapping:
        abnormal: 1
        normal: 0
      source: label
      target: label

lr_scheduler:
  select: CosineLRScheduler
  period_epochs: 50
  min_lr: 1e-6
  warmup_start_lr: 0
  warmup_epochs: 5

model:
  select: labram.labram_base_patch200_200
  finetune: /home/lingyus/code/PRL/models/labram/checkpoints/labram-base.pth
  nb_classes: 2
  use_mean_pooling: true

optimizer:
  lr: 5e-4
  select: AdamW
  weight_decay: 0.05
  layer_decay: 0.65

task:
  loss:
    select: LabelSmoothingCrossEntropy
    smoothing: 0.1
  select: tuab_task.TUABTask

trainer:
  fp16: true
  total_epochs: 50
  update_interval: 8
  log_interval: 20
  eval_metric:
    select: balanced_accuracy
    mode: max
  metrics: [balanced_accuracy, accuracy, pr_auc, roc_auc]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 和预训练模型路径 (`model.finetune`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/tuab.yaml
   ```

   *(请将 `config/tuab.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。该结果的评估方式为：最佳模型基于训练集训练，通过在验证集上取得的**最佳平衡准确率 (best balanced accuracy)** 进行选择，并最终在测试集上进行评估。

|      | Balanced Accuracy | AUC-PR | AUROC |
| ---- | ----------------- | ------ | ----- |
| Tyee | 82.37             | 90.96  | 90.65 |
| 官方 | 81.40             | 89.65  | 90.22 |