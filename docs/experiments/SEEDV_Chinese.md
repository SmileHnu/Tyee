# SEED-V - 情绪识别

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | SEED-V                                                       |
| **信号类型** | EEG-DE (脑电微分熵), EM-DE (眼动微分熵)                      |
| **分析任务** | 情绪识别 (Emotion Recognition)，5分类任务 (happy, sad, neutral, fear, disgust) |
| **使用模型** | G2G-ResNet18 (Graph-to-Grid with ResNet-18 backbone)         |
| **参考论文** | [Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://dl.acm.org/doi/abs/10.1145/3581783.3612074) |
| **原始代码** | https://github.com/Jinminbox/G2G                             |

本实验旨在使用 Tyee 框架，在 **SEED-V** 数据集的 **DE特征** 上，通过 **Graph-to-Grid (G2G)** 方法完成 **5分类情绪识别** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中处理图结构脑电数据和多模态特征融合的实用范例。

------

## 2. 准备工作

- **下载地址**：[SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html)

- **目录结构**：请将下载并解压数据集按原始结构存放即可，即：

  ```
  /path/to/data/SEED-V/
  ├── EEG_DE_features/
  │   ├── 1_123.npz
  │   ├── 2_123.npz
  │   └── ...
  └── Eye_movement_features/
  │   ├── 1_123.npz
  │   ├── 2_123.npz
  │   └── ...
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **G2G-ResNet18** 模型。该方法的核心思想是通过一个 `RelationAwareness` 模块，将基于电极物理位置的图结构脑电特征，动态地映射成具有空间拓扑信息的2D网格（图像）表示。随后，将生成的图像输入到经典的 **ResNet-18** 骨干网络中进行特征提取和分类。

### 3.2 模型参数

模型关键参数在配置文件中设置如下：

- **`head_num`**: 6 (关系感知模块中的注意力头数)
- **`rand_ali_num`**: 2 (随机排列分支的数量)
- **`backbone`**: "ResNet18" (图像处理的骨干网络)
- **`input_size`**: 5 (每个电极的EEG特征维度，即5个频带的DE值)
- **`location_size`**: 3 (电极的三维空间坐标维度)
- **`expand_size`**: 10 (特征和位置信息的嵌入扩展维度)
- **`num_class`**: 5 (最终分类的情绪类别数)

该模型在本实验中**从零开始 (from scratch)** 进行训练。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`KFoldPerSubjectCross`)**: 本实验采用**受试者内3折交叉验证**。对每个受试者的数据，按试验 (`trial_id`) 分为3折，轮流将其中1折作为验证集，其余2折作为训练集。

### 4.2 数据处理流程

由于使用的是官方提取好的特征，本实验的数据处理流程相对简单，主要在 `online_signal_transform` 部分定义：

1. **归一化 (`MinMaxNormalize`)**: 分别对 EEG 和 EOG 特征进行 Min-Max 归一化。
2. **特征合并 (`Concat`)**: 将归一化后的 EEG 和 EOG 特征合并。
3. **特征对齐 (`Insert`)**: 根据模型需要，在特定位置插入0值以对齐特征维度。
4. **维度压缩 (`Squeeze`)**: 移除多余的维度以匹配模型输入。

### 4.3 任务定义

- **任务类型**: `seedv_task.SEEDVFeatureTask`
- **核心逻辑**:
  - **优化器参数设置 (`set_optimizer_params`)**: 该方法对模型的不同部分（G2G模块、骨干网络、分类头）进行了参数分组，便于未来可能实现的差异化学习率训练。
  - **训练/验证步骤**: 接收处理后的多模态特征 `eeg_eog` 和情绪标签 `emotion`，通过模型进行前向传播，并使用带标签平滑的**交叉熵损失 (`LabelSmoothingCrossEntropy`)** 计算损失值。

### 4.4 训练策略

- **优化器**: 使用 **AdamW**，学习率为 **0.008**，权重衰减为 `5e-4`。
- **学习率调度器 (`StepLRScheduler`)**: 采用阶梯式学习率下降策略，并在开始时有2000步的线性预热。
- **训练周期**: 共训练 **300个 epoch**。
- **评估指标**: 使用**准确率 (`accuracy`)** 作为核心评估指标来选择最佳模型。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 222
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 32
  dataset: seedv_dataset.SEEDVFeatureDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/SEED-V'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_seedv_feature/train"
  io_mode: hdf5
  io_chunks: 1
  split: 
    select: KFoldPerSubjectCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_seedv_feature/split
      group_by: trial_id
      n_splits: 3
      shuffle: false

  offline_signal_transform:
    - select: Log
      epsilon: 1
      source: eog
      target: eog
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: eeg
      target: eeg
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: eog
      target: eog

  offline_label_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: emotion
      target: emotion

  online_signal_transform:
    - select: MinMaxNormalize
      source: eeg
      target: eeg
    - select: MinMaxNormalize
      source: eog
      target: eog
    - select: Concat
      axis: -1
      source: ["eeg", "eog"]
      target: eeg_eog
    - select: Insert
      indices: [316, 317, 318, 319, 326, 327, 328, 329, 334, 335, 336, 337, 338, 339, 344, 345, 346, 347, 348, 349, 354, 355, 356, 357, 358, 359, 369]
      value: 0
      axis: -1
      source: eeg_eog
      target: eeg_eog
    - select: Squeeze
      axis: 0
      source: eeg_eog
      target: eeg_eog
    - select: Select
      key: ["eeg_eog"]

  online_label_transform:
    - select: Squeeze
      axis: 0
      source: emotion
      target: emotion
    - select: ToNumpyInt64
      source: emotion
      target: emotion
  
lr_scheduler:
  select: StepLRScheduler
  gamma: 0.1
  epoch_size: 300
  warmup_steps: 2000 

model:
  select: g2g.EncoderNet
  head_num: 6
  rand_ali_num: 2
  backbone: "ResNet18"
  input_size: 5
  location_size: 3
  expand_size: 10
  eeg_node_num: 62
  num_class: 5
  sup_node_num: 6

optimizer:
  lr:  0.008
  select: AdamW
  weight_decay: 5e-4
  betas: (0.9, 0.999)

task:
  loss:
    select: LabelSmoothingCrossEntropy
    smoothing: 0.01
  select: seedv_task.SEEDVFeatureTask

trainer:
  fp16: false
  total_epochs: 300
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, balanced_accuracy,precision_macro, f1_macro, recall_macro]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/seedv.yaml
   ```

   *(请将 `config/seedv.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的验证性能。由于本实验采用受试者内3折交叉验证，下表展示的是**所有受试者及所有折上验证集结果的平均性能**。

|      | 准确率 (Accuracy) |
| ---- | ----------------- |
| Tyee | 77.05             |
| 官方 | 76.01             |

注：官方为原始代码复现结果