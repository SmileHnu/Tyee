# PPG-DaLiA 心率估计

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | PPG-DaLiA                                                    |
| **信号类型** | PPG (光电容积脉搏波), ACC (三轴加速度计)                     |
| **分析任务** | 心率估计 (Heart Rate Estimation)，一个从PPG和ACC信号中预测连续心率值的回归任务。 |
| **使用模型** | BeliefPPG                                                    |
| **参考论文** | [BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation](https://proceedings.mlr.press/v216/bieri23a.html) |
| **原始代码** | https://github.com/eth-siplab/BeliefPPG                      |

本实验旨在使用 Tyee 框架，在 **PPG-DaLiA** 数据集上，通过 **BeliefPPG** 模型完成 **基于PPG和ACC信号的心率估计** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中进行多模态回归和处理复杂任务逻辑的实用范例。

------

## 2. 准备工作

- **下载地址**：[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia)

- 目录结构：请将下载并解压的数据集 (`PPG_FieldStudy`文件夹) 按以下结构存放：

  ```
  /path/to/data/ppg_dalia/
  └── PPG_FieldStudy/
      ├── S1/
      ├── S2/
      └── ...
  ```

------

## 3. 模型配置

本实验选用 **BeliefPPG** 模型,将使用其默认的架构设置进行实例化。

------

## 4. 实验配置与数据处理

本实验的所有设置，包括复杂的数据处理流程、模型参数和训练策略，均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略 (`LosoRotatingCrossSplit`)**: 本实验采用一种轮换的留一法交叉验证。在每一“折” (fold) 的实验中，会选择 **1 位受试者作为测试集**，**2 位受试者作为验证集**，其余所有受试者的数据则用于训练。通过轮换不同的受试者组合，模型最终会在所有数据上进行训练和评估。

### 4.2 数据处理流程

本实验的数据预处理流程非常复杂，完全定义在配置文件的 `offline_signal_transform` 和 `offline_label_transform` 部分。其核心步骤包括：

1. **初级滑窗**：首先对原始的 PPG 和 ACC 信号进行第一次滑窗 (`SlideWindow`)，以提取重叠的信号片段。
2. **特征提取**: 在每个信号片段内，执行一系列变换，如去趋势、滤波、标准化，并最终通过快速傅里叶变换 (`FFTSpectrum`) 生成时频谱特征。
3. **特征堆叠与二次滑窗**: 将 PPG 和 ACC 的时频谱特征堆叠 (`Stack`) 起来，并对这个新的特征序列进行第二次滑窗，以捕捉时序频谱动态。
4. **时域特征提取**: 同时，从原始 PPG 信号中也提取另一路时域特征。
5. 最终，模型接收处理后的**时频谱序列 (`ppg_acc`)** 和**时域序列 (`ppg_time`)** 作为双路输入。

### 4.3 任务定义

- **任务类型**: `dalia_hr_task.DaLiaHREstimationTask`
- 核心逻辑:
  - **拟合先验层 (`on_train_start`)**: 在训练开始前，通过一个钩子函数遍历全部训练数据，让模型的 `PriorLayer` 学习并拟合训练集中所有心率标签的先验分布。
  - **损失函数 (`BinnedRegressionLoss`)**: 使用一种特殊的“分箱回归损失”，将连续值预测问题转化为对概率分布的预测。
  - **验证步骤 (`valid_step`)**: 在验证时，会激活模型的 `PriorLayer` 来结合先验知识修正模型的输出。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **2.5e-4**。
- **学习率调度器 (`ReduceLROnPlateauScheduler`)**: 监控训练集损失，若连续3个epoch无下降则学习率减半。
- **评估指标**: 使用**平均绝对误差 (Mean Absolute Error, `mae`)** 作为核心评估指标，选择 `mae` 最低的模型为最佳模型。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: dalia_dataset.DaLiADataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/ppg_dalia/PPG_FieldStudy'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_ppgdalia/train"
  io_mode: hdf5
  io_chunks: 320
  split: 
    select: LosoRotatingCrossSplit
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_ppgdalia/split_official
      n_splits: 4
      group_by: subject_id
      shuffle: true
      random_state: 7

  offline_signal_transform:
    - select: Compose
      transforms:
        - select: SlideWindow
          window_size: 512
          stride: 128
        - select: WindowExtract
        - select: ForEach
          transforms:
            - select: Detrend
            - select: Filter
              l_freq: 0.4
              h_freq: 4
              method: iir
              phase: forward
              iir_params:
                order: 4
                ftype: butter
            - select: ZScoreNormalize
              axis: -1
              epsilon: 1e-10
            - select: Mean
              axis: 0
            - select: Resample
              desired_freq: 25
              window: boxcar
              pad: constant
              npad: 0
            - select: FFTSpectrum
              resolution: 535
              min_hz: 0.5
              max_hz: 3.5
      source: ppg
      target: ppg_spec
    - select: Compose
      transforms:
        - select: SlideWindow
          window_size: 256
          stride: 64
        - select: WindowExtract
        - select: ForEach
          transforms:
            - select: Detrend
            - select: Filter
              l_freq: 0.4
              h_freq: 4
              method: iir
              phase: forward
              iir_params:
                order: 4
                ftype: butter
            - select: ZScoreNormalize
              axis: -1
              epsilon: 1e-10
            - select: Resample
              desired_freq: 25
              window: boxcar
              pad: constant
              npad: 0
            - select: FFTSpectrum
              resolution: 535
              min_hz: 0.5
              max_hz: 3.5
              axis: -1
            - select: Mean
              axis: 0
      source: acc
      target: acc_spec
    - select: Stack
      axis: -1
      source: [ppg_spec, acc_spec]
      target: ppg_acc
    - select: Compose
      transforms:
        - select: ZScoreNormalize
          epsilon: 1e-10
        - select: SlideWindow
          window_size: 7
          stride: 1
          axis: 0
      source: ppg_acc
      target: ppg_acc
    - select: Compose
      transforms:
        - select: Detrend
        - select: Filter
          l_freq: 0.1
          h_freq: 18
          method: iir
          phase: forward
          iir_params:
            order: 4
            ftype: butter
        - select: Mean
          axis: 0
        - select: ExpandDims
          axis: -1
        - select: ZScoreNormalize
          epsilon: 1e-10
        - select: SlideWindow
          window_size: 1280
          stride: 128
          axis: 0
      source: ppg
      target: ppg_time
    - select: Select
      key: [ppg_acc, ppg_time]

  offline_label_transform:
    - select: Compose
      transforms:
        - select: Crop
          crop_left: 6
        - select: SlideWindow
          window_size: 1
          stride: 1
          axis: 0
      source: hr
      target: hr

lr_scheduler:
  select: ReduceLROnPlateauScheduler
  patience_epochs: 3
  factor: 0.5
  min_lr: 1e-6
  metric_source: train
  metric: loss

model:
  select: beliefppg.beliefppg.BeliefPPG

optimizer:
  lr: 2.5e-4
  select: Adam

task:
  loss:
    select: BinnedRegressionLoss
    dim: 64
    min_hz: 0.5
    max_hz: 3.5
    sigma_y: 1.5
  select: dalia_hr_task.DaLiaHREstimationTask

trainer:
  fp16: true
  total_epochs: 50
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: mae
    mode: min
  metrics: [mae, r2]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`, `split_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```bash
   python main.py --config config/dalia.yaml
   ```

   *(请将 `config/dalia.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。该结果的评估方式为：在交叉验证的**每一折**中，我们选出在验证集上取得**最低平均绝对误差 (best MAE)** 的最佳模型，并在该折对应的测试集上进行评估。下表展示的是**所有折上测试结果的平均性能**。

|      | 平均绝对误差 (MAE) |
| ---- | ------------------ |
| Tyee | 4.22               |
| 官方 | 4.02               |

**注**：原 BeliefPPG 论文的实验中采用了数据增强，但本复现实验并未采用。