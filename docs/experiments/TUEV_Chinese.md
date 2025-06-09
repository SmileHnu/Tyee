# TUEV - 事件分类

## 1. 实验概述 

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | TUH EEG Event Corpus (TUEV) v2.0.1                           |
| **信号类型** | EEG                                                          |
| **分析任务** | 事件分类 (Event Classification)，6分类任务 (eyem, chew, shiw, musc, elpp, null) |
| **使用模型** | LaBraM (Large Brain Model)                                   |
| **参考论文** | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) |
| **原始代码** | https://github.com/935963004/LaBraM                          |

本实验旨在使用 Tyee 框架，在 **TUEV** 数据集上，通过微调 (fine-tuning) **LaBraM** 预训练模型来完成 **脑电事件六分类** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中使用和微调大型预训练模型的实用范例。

------

## 2. 准备工作

- **下载地址**：[TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuev)(需要申请和授权)

- 目录结构：请将下载并解压的数据集 (`v2.0.1/edf/`) 按以下结构存放：

  ```
  /path/to/data/tuh_eeg_events/v2.0.1/edf/
  ├── train/
  │   ├── 000/
  │   └── ...
  └── eval/
      ├── 000/
      └── ...
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **LaBraM** (`labram_base_patch200_200`) 模型进行异常脑电检测。

### 3.2 模型参数

该模型在本实验中的关键参数在配置文件中设置如下：

- **`nb_classes`**: 6 (对应6分类)
- **`drop`**: 0.0 (全连接层中的 Dropout 率)
- **`drop_path`**: 0.1 (Stochastic Depth 的 Drop Path 率)
- **`attn_drop_rate`**: 0.0 (注意力机制中的 Dropout 率)
- **`use_mean_pooling`**: true (使用平均池化来聚合特征)
- **`layer_scale_init_value`**: 0.1 (层缩放（Layer Scale）的初始值)

### 3.3 预训练权重说明

在本实验中，我们加载了 **LaBraM** 的**官方上游预训练权重**作为模型初始化的起点，并在 TUEV 数据集上进行微调。

- **权重来源**：该预训练权重由模型原作者发布。
- **下载链接**：https://github.com/935963004/LaBraM/tree/main/checkpoints
- **使用方法**：请将下载的权重文件放置于配置文件中 `model.finetune` 指定的路径下，Tyee 将在实验开始时自动加载此权重。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

本实验遵循 TUEV 数据集官方的训练集和测试集划分。在此基础上，我们将官方的**训练集**进一步划分为新的训练集和验证集，用于模型选择和监控。

- **划分策略 (`HoldOutCross`)**: 从原始训练集中按受试者ID (

  `group_by: subject_id`) 进行划分，以确保同一受试者的数据不会同时出现在训练集和验证集中。 

- **划分比例 (`val_size`)**: 将 20% 的受试者划入验证集，其余 80% 作为新的训练集。 

### 4.2 数据处理流程

我们对数据应用了多个阶段的处理：

1. 分段前处理 (`before_segment_transform`)：
   - **通道选择 (`PickChannels`)**：挑选出标准的 23 个 EEG 通道。
   - **滤波 (`Filter` & `NotchFilter`)**：应用 0.1-75 Hz 的带通滤波，并使用 50 Hz 的陷波器去除工频干扰。
   - **重采样 (`Resample`)**：将信号的采样率统一降至 200 Hz。
2. 离线处理 (`offline_signal_transform`)：
   - **滑窗 (`SlideWindow`)**：将连续信号以 1000 个时间点（5秒）为窗口大小、1000 为步长进行切分，无重叠。
3. 在线标签处理 (`online_label_transform`)：
   - **偏移 (`Offset`)**：将标签值减 1，以匹配从0开始的类别索引。 

### 4.3 任务定义

- **任务类型**: `tuev_task.TUEVTask`
- 核心逻辑: 
  - **模型构建 (`build_model`)**: 负责加载 LaBraM 模型结构，并从指定路径加载官方预训练权重。
  - **优化器参数设置 (`set_optimizer_params`)**: 实现了**层级学习率衰减 (Layer-wise Learning Rate Decay)** 策略，靠近输出层的学习率较大，而靠近输入的底层学习率较小。
  - **训练/验证步骤 (`train_step`/`valid_step`)**: 在将数据送入模型前，会进行特定的 `rearrange` 操作以匹配 LaBraM 的输入格式，并对输入信号进行 `/100` 的缩放。

### 4.4 训练策略

- **优化器**: 使用 AdamW，并为其设置了 0.05的权重衰减（`weight_decay`）和 0.65 的层级学习率衰减（`layer_decay`）。 
- **学习率调度器**: 采用 余弦退火学习率 (`CosineLRScheduler`)，总周期为50个epoch，并包含前5个epoch的线性预热（warm-up）阶段。 
- **损失函数**: 采用带标签平滑（smoothing=0.1）的交叉熵损失 (`LabelSmoothingCrossEntropy`)，有助于提升模型的泛化能力。 
- **梯度累积**: `update_interval`设置为8，表示每累积8个batch的梯度后进行一次模型参数更新，等效于增大了批处理大小。 

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: tuev_dataset.TUEVDataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/tuh_eeg_events/v2.0.1/edf/train'
    test: '/mnt/ssd/lingyus/tuh_eeg_events/v2.0.1/edf/eval'
  io_path:
    train: "/mnt/ssd/lingyus/tuh_eeg_events/v2.0.1/edf/processed_train_yaml"
    test: "/mnt/ssd/lingyus/tuh_eeg_events/v2.0.1/edf/processed_eval_yaml"
  split: 
    select: HoldOutCross
    init_params:
      split_path: /mnt/ssd/lingyus/tuh_eeg_events/v2.0.1/edf/split
      group_by: subject_id
      val_size: 0.2
      random_state: 4523
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
      window_size: 1000
      stride: 1000
      source: eeg
      target: eeg
  
  online_label_transform:
    - select: Offset
      offset: -1
      source: event
      target: event

lr_scheduler:
  select: CosineLRScheduler
  period_epochs: 50
  min_lr: 1e-6
  warmup_start_lr: 0
  warmup_epochs: 5

model:
  select: labram.labram_base_patch200_200
  finetune: /home/lingyus/code/PRL/models/labram/checkpoints/labram-base.pth
  nb_classes: 6
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
  select: tuev_task.TUEVTask

trainer:
  fp16: true
  total_epochs: 50
  update_interval: 8
  log_interval: 20
  eval_metric:
    select: balanced_accuracy
    mode: max
  metrics: [balanced_accuracy, accuracy, f1_weighted, cohen_kappa]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 和预训练模型路径 (`model.finetune`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```yaml
   python main.py --config config/tuev.yaml
   ```

   *(请将 `config/tuev.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹，例如：`./experiments/2025-06-07/11-45-00/`。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。该结果的评估方式为：最佳模型基于训练集训练，通过在验证集上取得的**最佳平衡准确率 (best balanced accuracy)**  进行选择，并最终在测试集上进行评估。

|          | 平衡准确率 (Balanced Accuracy) | F1分数 (Weighted) | Cohen's Kappa |
| -------- | ------------------------------ | ----------------- | ------------- |
| Tyee     | 64.78                          | 82.89             | 65.50         |
| **官方** | 64.09                          | 83.12             | 66.37         |

