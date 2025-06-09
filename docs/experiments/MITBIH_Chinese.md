# MIT-BIH - 心律失常分类

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | MIT-BIH Arrhythmia Database                                  |
| **信号类型** | ECG (心电图)                                                 |
| **分析任务** | 心律失常分类 (Arrhythmia Classification)，一个旨在区分不同心跳类型的多分类任务。 |
| **使用模型** | EcgResNet34                                                  |
| **参考论文** | [Diagnosis of Diseases by ECG Using Convolutional Neural Networks](https://www.hse.ru/en/edu/vkr/368722189) |
| **原始代码** | https://github.com/lxdv/ecg-classification                   |

本实验旨在使用 Tyee 框架，在 **MIT-BIH** 数据集上，通过一个定制的 **EcgResNet34** 模型完成 **心律失常分类** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中进行ECG信号分类的实用范例。

------

## 2. 准备工作

- **下载地址**：[MIT-BIH Arrhythmia Database on PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

- 目录结构：请将下载并解压的数据集按以下结构存放：

  ```
  /path/to/data/mit-bih-arrhythmia-database-1.0.0/
  ├── 100.atr
  ├── 100.dat
  ├── 100.hea
  └── ...
  ```

------

## 3. 模型配置

本实验选用 **EcgResNet34** 模型，将使用其默认的架构设置进行实例化。

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`HoldOut`)**: 将所有记录中的全部心跳样本混合后，进行**分层随机划分** (`stratify: symbol`)。分层策略确保了在训练集和验证集中，不同心律类型的样本比例与原始数据集大致相同。
- **划分比例 (`val_size`)**: 将 **10%** 的数据划入验证集，其余 **90%** 作为训练集。

### 4.2 数据处理流程

本实验的数据预处理流程完全定义在配置文件的 `before_segment_transform` 和 `offline_label_transform` 部分。其核心步骤包括：

1. 信号 (ECG) 处理:
   - **通道选择 (`PickChannels`)**: 仅选择 'MLII' 导联的信号进行分析。
   - **标准化 (`ZScoreNormalize`)**: 对信号进行 Z-score 标准化。
2. 标签 (Symbol) 处理:
   - **标签映射 (`Mapping`)**: 将 'N', 'V' 等心律类型的字符标签映射为 0, 1, 2... 等数字类别索引。

### 4.3 任务定义

- **任务类型**: `mit_bih_task.MITBIHTask`
- **核心逻辑**: 该任务负责接收 ECG 信号 `x` 和心律类型标签 `symbol`，通过 EcgResNet34 模型进行前向传播，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失值以驱动模型训练。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **0.001**。
- **训练周期**: 共训练 **650个 epoch**。
- **评估指标**: 使用**准确率 (`accuracy`)** 作为核心评估指标，选择 `accuracy` 最高的模型为最佳模型。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: mit_bih_dataset.MITBIHDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/mit-bih-arrhythmia-database-1.0.0'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_mit_bih/train"
  io_mode: hdf5
  io_chunks: 128
  split: 
    select: HoldOut
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_mit_bih/split
      val_size: 0.1
      random_state: 7
      shuffle: true
      stratify: symbol

  before_segment_transform:
    - select: PickChannels
      channels: ['MLII']
      source: ecg
      target: ecg
    - select: ZScoreNormalize
      axis: 1
      source: ecg
      target: ecg
      
  offline_label_transform:
    - select: Mapping
      mapping:
        'N': 0
        'V': 1
        '/': 2
        'R': 3
        'L': 4
        'A': 5
        '!': 6
        'E': 7
      source: symbol
      target: symbol

model:
  select: ecgresnet34.EcgResNet34

optimizer:
  lr:  0.001
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: mit_bih_task.MITBIHTask

trainer:
  fp16: false
  total_epochs: 650
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

   ```bash
   python main.py --config config/mit_bih.yaml
   ```

   *(请将 `config/mit_bih.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的验证性能。该结果的评估方式为：最佳模型基于训练集训练，通过在验证集上取得的**最佳准确率 (best accuracy)** 进行选择。

|      | 准确率 (Accuracy) |
| ---- | ----------------- |
| Tyee | 99.51             |
| 官方 | 99.38             |