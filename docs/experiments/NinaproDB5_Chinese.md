# Ninapro DB5 - sEMG手势识别

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | Ninapro Database 5 (Exercise 2)                              |
| **信号类型** | sEMG (表面肌电)                                              |
| **分析任务** | 手势识别 (Gesture Recognition)，一个旨在从sEMG信号中区分10种不同手部姿态的多分类任务。 |
| **使用模型** | ResNet-18                                                    |
| **参考论文** | [EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html) |
| **原始代码** | (https://github.com/jehanyang/emgbench                       |

本实验旨在使用 Tyee 框架，在 **Ninapro DB5** 数据集上，通过一个包含**预训练**和**个性化微调**的复杂流程，完成 **sEMG手势识别** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果。

------

## 2. 准备工作

- **下载地址**：[NinaproDB5](https://ninapro.hevs.ch/instructions/DB5.html)

- 目录结构：该实验只在Exercise 2上进行，请将下载并解压的数据集按以下结构存放：

  ```
  /path/to/data/NinaproDB5E2/
  ├── S1_E2_A.mat
  ├── S2_E2_A.mat
  └── ...
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用经典的 **ResNet-18** 模型。为了处理一维的 sEMG 信号，我们先将其转换为二维图像（频谱图或类似形式），然后输入到为图像识别设计的 ResNet-18 中。该模型通过 `timm` 库进行加载。

### 3.2 预训练权重说明

在本实验中，ResNet-18 的骨干网络首先会加载在 **ImageNet** 上预训练的权重，然后在我们的**第一阶段预训练**中进一步更新。

- **权重来源**：ImageNet (由`timm`库自动处理)
- **使用方法**：在配置文件中设置 `model.pretrained: true` 即可。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

本实验的划分策略非常独特，遵循 EMGBench 论文中的方法，分为两个嵌套的层次，旨在模拟对新用户（未见过的受试者）进行个性化适应的场景：

1. 第一层划分 (预训练阶段 - `LeaveOneOut`):
   - **策略**: 采用**留一交叉验证 (Leave-One-Subject-Out)**。在每一“折” (fold) 中，会选择 **1 位受试者作为“留出受试者”**，其全部数据暂时不用；其余所有受试者的数据则全部作为**预训练阶段的训练集**。
2. 第二层划分 (微调阶段 - `HoldOutPerSubject`):
   - **策略**: 针对第一层中被留出的那位受试者的数据，进行内部的**留出法划分**。
   - 划分比例:
     - **20%** 的数据用作**微调训练集 (fine-tuning training set)**。
     - **40%** 的数据用作**验证集 (validation set)**。
     - **40%** 的数据用作**测试集 (test set)**。
   - 所有划分都在手势类别上进行了**分层 (`stratify: gesture`)**，以保证数据分布的均衡。

### 4.2 数据处理流程

本实验的数据预处理流程在配置文件中定义，核心步骤是将一维sEMG时序信号转换为二维图像：

1. **信号滤波 (`Filter`)**: 对原始sEMG信号应用一个3阶的巴特沃斯带通滤波器。
2. **信号塑形 (`Reshape`)**: 将滤波后的信号片段重塑为 `16x50` 的二维矩阵。
3. **图像转换 (`ToImage`)**: 将该二维矩阵转换为 `viridis` 色彩映射的图像。
4. **尺寸调整 (`ImageResize`)**: 在线将生成的图像尺寸调整为 `224x224`，以匹配 ResNet-18 的标准输入。

### 4.3 任务定义

- **任务类型**: `ninapro_db5_task.NinaproDB5Task`
- **核心逻辑**: 该任务负责接收处理后的 sEMG 图像 `emg` 和手势标签 `gesture`，通过 ResNet-18 模型进行前向传播，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失值以驱动模型训练。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **5e-4**。
- **训练周期**: 预训练阶段为**100个epoch**，微调阶段为 **750个 epoch**。
- **评估指标**: 使用**准确率 (`accuracy`)** 作为核心评估指标，选择 `accuracy` 最高的模型为最佳模型。

### 4.5 完整配置文件

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: ninapro_db5_dataset.NinaproDB5Dataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/NinaproDB5E2'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_ninapro_db5/train"
  io_mode: hdf5
  split: 
  	# 预训练阶段的划分配置
    # select: LeaveOneOutAndHoldOutET
    # init_params:
    #   split_path: /mnt/ssd/lingyus/tyee_ninapro_db5/split_pretrain
    #   stratify: gesture
    #   shuffle: false
    #   test_size: 0.4
    #   group_by: subject_id
    
    # 微调阶段的划分配置
    select: HoldOutPerSubjectET
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_ninapro_db5/split_finetune
      stratify: gesture
      shuffle: false
      val_size: 0.4
      test_size: 0.4
    run_params:
      subject: 3 # 示例：对第3位受试者进行微调和测试

  offline_label_transform:
    - select: Mapping
      mapping: {0: 0, 17: 1, 18: 2, 20: 3, 21: 4, 22: 5, 25: 6, 26: 7, 27: 8, 28: 9}
      source: gesture
      target: gesture
    - select: OneHotEncode
      num: 10
      source: gesture
      target: gesture

  offline_signal_transform:
    - select: Filter
      l_freq: 5.0
      method: iir
      iir_params: {order: 3, ftype: butter, padlen: 12}
      phase: zero
      source: emg
      target: emg
    - select: Reshape
      shape: 800
      source: emg
      target: emg
    - select: ToImage
      length: 16
      width: 50
      cmap: viridis
      source: emg
      target: emg
    - select: ToNumpyFloat16
      source: emg
      target: emg
  
  online_signal_transform:
    - select: ImageResize
      size: [224, 224]
      source: emg
      target: emg

model:
  select: resnet18.resnet18
  num_classes: 10
  # ImageNet预训练权重
  pretrained: true
  # ImageNet预训练权重可自行下载并保存到指定路径，通过下面方式加载
  #pretrained_cfg_overlay:
  #  file: '/home/lingyus/code/emgbench/checkpoint/resnet18/pytorch_model.bin'
  
  # 微调阶段加载第一阶段预训练好的模型
  checkpoint_path: /home/lingyus/code/PRL/experiments/2025-05-23/11-24-05-ninapro_db5_task.NinaproDB5Task/checkpoint/fold_3/checkpoint_best.pt

optimizer:
  lr:  5e-4
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: ninapro_db5_task.NinaproDB5Task

trainer:
  fp16: false
  total_epochs: 750
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, balanced_accuracy]
```

------

## 5. 复现步骤

本实验包含两个阶段：

1. **第一阶段：预训练**

   - 首先，需要配置并运行一个预训练流程。这通常涉及修改 `config.yaml` 中的 `split` 部分，选择 `LeaveOneOut` 类似的策略，在除留出受试者之外的所有数据上进行训练。
   - 保存为每个留出受试者训练好的模型检查点。

2. **第二阶段：个性化微调与评估**

   - **确认配置文件**：使用上文提供的 `config.yaml`。确保所有数据路径 (`root_path`, `io_path`, `split_path`) 以及第一阶段预训练好的模型路径 (`model.checkpoint_path`) 都已正确设置。

   - **执行实验**：在项目根目录下，运行以下命令。该命令将加载预训练模型，并在每个留出受试者的数据上进行微调和评估。

     ```bash
     python main.py --config config/ninapro_db5_finetune.yaml
     ```

     *(请将 `config/ninapro_db5_finetune.yaml` 替换为你的实际配置文件路径)*

   - **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。由于本实验采用留一交叉验证，下表中的数据代表了**所有受试者（折）的平均测试性能**。在每一折中，最佳模型都是根据其在验证集上取得的**最佳准确率 (best accuracy)** 筛选得出，并最终在相应的测试集上进行评估。

|      | 准确率 (Accuracy) |
| ---- | ----------------- |
| Tyee | 71.1              |
| 官方 | 68.3              |

