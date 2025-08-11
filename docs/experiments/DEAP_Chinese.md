# DEAP - 情感识别

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | DEAP                                                         |
| **信号类型** | GSR, Respiration, PPG, Temperature                           |
| **分析任务** | 情感识别 - 唤醒度 (Arousal) 分类，9分类任务                  |
| **使用模型** | MLSTM-FCN                                                    |
| **参考论文** | [Multivariate LSTM-FCNs for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200) |
| **原始代码** | https://github.com/titu1994/MLSTM-FCN（模型原始仓库）<br />https://github.com/athar70/MLSTM（DEAP实验） |

本实验旨在使用 Tyee 框架，在 **DEAP** 数据集上，通过 **MLSTM-FCN** 模型完成 **基于多模态外周生理信号的唤醒度9分类** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中进行多模态时间序列分类的实用范例。

------

## 2. 准备工作

- **下载地址**：[DEAP](http://eecs.qmul.ac.uk/mmv/datasets/deap/)

- 目录结构：请将下载并解压的数据集 (`data_preprocessed_python`文件夹) 按以下结构存放：

  ```
  /path/to/data/DEAP/
  └── data_preprocessed_python/
      ├── s01.dat
      ├── s02.dat
      └── ...
  ```

------

## 3. 模型配置

本实验选用 **MLSTM-FCN** 模型进行唤醒度分类。其关键参数在配置文件中设置如下：

- **`max_nb_variables`**: 4 (对应4种生理信号：GSR, Resp, PPG, Temp)
- **`max_timesteps`**: 640 (每个样本的时间点数)
- **`nb_class`**: 9 (对应9个唤醒度等级)

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`HoldOut`)**: 将全部32名受试者的所有试验数据混合后，进行随机划分。
- **划分比例 (`val_size`)**: 将 **30%** 的数据划入验证集，其余 **70%** 作为训练集。

### 4.2 数据处理流程

本实验的数据预处理流程完全定义在配置文件的 `offline_signal_transform` 和 `offline_label_transform` 部分。其核心步骤包括：

1. 信号 (Signal) 处理:
   - **通道合并 (`Concat`)**: 将 GSR, Resp, PPG, Temp 四种信号在通道维度上合并。
   - **归一化 (`MinMaxNormalize`)**: 对合并后的多模态信号进行 Min-Max 归一化。
   - **滑窗 (`SlideWindow`)**: 使用大小为 640、步长为 384 的窗口对信号进行切分，以生成样本。
2. 标签 (Label) 处理:
   - **数值处理 (`Round`, `ToNumpyInt32`)**: 对原始浮点型标签进行四舍五入并转为整数。
   - **标签映射 (`Mapping`)**: 将 1-9 的原始标签值映射为 0-8 的类别索引。

### 4.3 任务定义

- **任务类型**: `deap_task.DEAPTask`
- **核心逻辑**: 该任务负责接收合并后的多模态信号 `mulit4` 和处理后的唤醒度标签 `arousal`，通过模型进行前向传播，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失值以驱动模型训练。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **1e-3**。
- **学习率调度器 (`ReduceLROnPlateauScheduler`)**: 监控训练集损失，若连续100个epoch无下降则学习率乘以约0.7。
- **训练周期**: 共训练 **2000个 epoch**。
- **评估指标**: 使用**准确率 (`accuracy`)** 作为核心评估指标，选择 `accuracy` 最高的模型为最佳模型。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: deap_dataset.DEAPDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/DEAP/data_preprocessed_python'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_deap/train"
  io_mode: hdf5
  io_chunks: 640
  split: 
    select: HoldOut
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_deap/split
      val_size: 0.3
      random_state: 42
      shuffle: true
      
  offline_signal_transform:
    - select: Concat
      axis: 0
      source: ['gsr', 'resp', 'ppg', 'temp']
      target: mulit4
    - select: Compose
      transforms:
        - select: MinMaxNormalize
          axis: -1
        - select: SlideWindow
          window_size: 640
          stride: 384
      source: mulit4
      target: mulit4
    - select: Select
      key: ['mulit4']

  offline_label_transform:
    - select: Compose
      transforms:
        - select: Round
        - select: ToNumpyInt32
        - select: Mapping
          mapping:
            1: 0
            2: 1
            3: 2
            4: 3
            5: 4
            6: 5
            7: 6
            8: 7
            9: 8
      source: arousal
      target: arousal
    - select: Select
      key: ['arousal']

lr_scheduler:
  select: ReduceLROnPlateauScheduler
  patience_epochs: 100
  factor: 0.7071
  min_lr: 1e-4
  metric_source: train
  metric: loss

model:
  select: mlstm_fcn.MLSTM_FCN
  max_nb_variables: 4
  max_timesteps: 640
  nb_class: 9

optimizer:
  lr:  1e-3
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: deap_task.DEAPTask

trainer:
  fp16: false
  total_epochs: 2000
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

2. 执行实验：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/deap.yaml
   ```

   (请将 `config/deap.yaml` 替换为你的实际配置文件路径)

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的验证性能。该结果的评估方式为：最佳模型基于训练集训练，通过在验证集上取得的**最佳准确率 (best accuracy)** 进行选择。

|      | 准确率 (Accuracy) |
| ---- | ----------------- |
| Tyee | 59.82             |
| 官方 | 47.3              |

### 结果分析

Tyee实现相比官方结果表现出显著更好的性能（准确率提升+12.52%）。这种改进可以归因于以下因素：

**复现方法说明**：在我们的复现过程中，我们遇到了原始官方代码库的技术问题，导致其无法正常运行。因此，我们采取了以下方法：

1. **参考官方结果**：我们使用原始仓库中报告的准确率（47.3%）作为基准。
2. **实现等效处理**：我们在Tyee框架中重新创建了相同的数据处理流程和模型架构，确保与原始方法在方法学上的一致性。
3. **框架优化**：Tyee实现可能受益于优化的训练程序。
