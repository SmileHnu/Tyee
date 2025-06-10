# CinC2018 - 睡眠分期

## 1. 实验概述

| 项目         | 描述                                                         |
| ------------ | ------------------------------------------------------------ |
| **数据集**   | PhysioNet Challenge 2018 (CinC2018)                          |
| **信号类型** | EEG, EOG, ECG, Chest Respiration, Abdominal Respiration, SaO2 |
| **分析任务** | 睡眠分期 (Sleep Staging)，5分类任务 (Wake, N1, N2, N3, REM)  |
| **使用模型** | SleepFM                                                      |
| **参考论文** | [SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity,ECG and Respiratory Signals](https://arxiv.org/abs/2405.17766) |
| **原始代码** | https://github.com/rthapa84/sleepfm-codebase                 |

本实验旨在使用 Tyee 框架，在 **CinC2018** 数据集上，通过微调 (fine-tuning) **SleepFM** 多模态预训练模型来完成 **睡眠分期** 任务。本页面详细记录了复现该实验所需的全部步骤、配置文件和预期结果，可作为在 Tyee 中应用大型多模态预训练模型的实用范例。

------

## 2. 准备工作

- **下载地址**：[CinC2018](https://physionet.org/content/challenge-2018/1.0.0/)

- 目录结构：请将下载并解压的数据集，按照 

  [4.1节](#41-数据集划分) 所述的划分方法，手动整理成 `train`, `valid`, `test`三个独立的文件夹。最终目录结构应如下：

  ```
  /path/to/data/challenge-2018-split/
  ├── train/
  │   ├── tr03-0001/
  │   └── ...
  ├── valid/
  │   ├── tr03-0002/
  │   └── ...
  └── test/
      ├── tr03-0004/
      └── ...
  ```

------

## 3. 模型配置

### 3.1 模型选择

本实验选用 **SleepFM** 模型。SleepFM 是一个在海量睡眠数据上进行自监督预训练的多模态基础模型，它包含三个并行的 EffNet 编码器，分别用于处理脑电活动、心电和呼吸信号，非常适合进行复杂的睡眠分期任务。

### 3.2 模型参数

该模型在本实验中的关键参数在配置文件中设置如下：

- **`num_classes`**: 5 (下游分类任务的类别数，即5个睡眠阶段)
- **`bas_in_channels`**: 5 (脑电活动模态的输入通道数)
- **`ecg_in_channels`**: 1 (心电模态的输入通道数)
- **`resp_in_channels`**: 3 (呼吸模态的输入通道数)
- **`embedding_dim`**: 512 (每个编码器输出的嵌入特征维度)
- **`freeze_encoders`**: true (冻结三个上游编码器的参数，仅训练下游分类头)
- **`effnet_depth`**: `[1, 2, 2, 3, 3, 3, 3]` (EffNet编码器中每个阶段的模块数量)
- **`effnet_expansion`**: 6 (EffNet中MBConv模块的扩展因子)

### 3.3 预训练权重说明

在本实验中，我们加载了 SleepFM 的官方上游预训练权重作为三个编码器的初始化起点，并根据 `freeze_encoders: true` 的设置冻结它们的参数，仅对下游分类头进行训练。

- **权重来源**：该预训练权重由模型原作者发布。

- **下载链接**：[SleepFM Checkpoints](https://github.com/rthapa84/sleepfm-codebase/tree/main/sleepfm/checkpoint)
- **使用方法**：请将下载的权重文件放置于配置文件中 `model.pretrained_checkpoint_path` 指定的路径下，Tyee 将在实验开始时自动加载此权重。

### 3.4 下游任务分类器

原 SleepFM 代码的下游任务使用了 `sklearn` 的逻辑回归分类器。由于 Tyee 是一个端到端的 PyTorch 框架，我们在此实验中**使用一个单层的 `nn.Linear` 线性层**来替代原有的逻辑回归，作为连接多模态融合特征和最终分类结果的分类头。

------

## 4. 实验配置与数据处理

本实验的所有设置均由唯一的配置文件 `config.yaml` 集中管理。

### 4.1 数据集划分

- **划分策略**: 我们严格遵循 SleepFM 原论文代码中的数据集划分方式。原作者将整个数据集划分为**预训练集 (75%)、训练集、验证集和测试集**。
- **实现方式**: 由于本实验仅进行下游任务微调，无需使用庞大的预训练集，我们直接采用了原作者划分好的**训练集、验证集和测试集**的受试者名单（详情见[**附录**](#附录数据集划分详情)）。您需要根据这份名单，手动从原始数据集中抽取出对应的受试者文件，并分别存放在 `train`, `valid`, `test` 三个文件夹中。因此，在配置文件中，我们为 `root_path` 和 `io_path` 分别提供了这三个路径，并使用 `NoSplit` 策略，表示直接使用预先分好的数据。

### 4.2 数据处理流程

我们对数据应用了多个阶段的处理：

1. 分段前处理 (`before_segment_transform`):
   - **通道选择/合并 (`PickChannels`, `Concat`)**: 从原始多通道EEG中挑选出4个通道，并与EOG合并为脑电活动信号(ss)；将胸腔、腹腔呼吸和血氧信号合并为呼吸信号(resp)。
   - **重采样 (`Resample`)**: 将所有模态的信号采样率统一为 256 Hz。
2. 离线处理 (`offline_signal_transform`):
   - **滑窗 (`SlideWindow`)**: 将每个记录切分为独立的30秒窗口。
3. 在线处理 (`online_signal_transform`):
   - **维度压缩 (`Squeeze`)**: 移除多余的维度以匹配模型输入。

### 4.3 任务定义

- **任务类型**: `cinc2018_task.CinC2018Task`
- 核心逻辑:
  - **优化器参数设置 (`set_optimizer_params`)**: 由于编码器被冻结，该方法被定制为**仅返回下游 `nn.Linear` 线性层的参数**进行优化。
  - **训练/验证步骤**: 接收三个模态的输入 `ss`, `ecg`, `resp`，通过 SleepFM 模型得到预测值 `pred`，并使用**交叉熵损失 (`CrossEntropyLoss`)** 计算损失。

### 4.4 训练策略

- **优化器**: 使用 **Adam**，学习率为 **0.001**，权重衰减为 `1e-4`。
- **训练周期**: 共训练 **100个 epoch**。
- **评估指标**: 使用**准确率 (`accuracy`)** 作为核心评估指标来选择最佳模型，同时计算包括 `f1_macro`, `roc_auc_macro_ovr` 在内的多种详细指标。

### 4.5 完整配置文件

以下是本实验使用的完整配置：

```yaml
common:
  seed: 42
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: cinc2018_dataset.CinC2018Dataset
  num_workers: 16
  root_path:
    train: '/mnt/ssd/lingyus/challenge-2018-split/train'
    val: '/mnt/ssd/lingyus/challenge-2018-split/valid'
    test: '/mnt/ssd/lingyus/challenge-2018-split/test' 
  io_path:
    train: "/mnt/ssd/lingyus/tyee_cinc2018/train"
    val: "/mnt/ssd/lingyus/tyee_cinc2018/valid"
    test: "/mnt/ssd/lingyus/tyee_cinc2018/test"
  io_mode: hdf5
  io_chunks: 1
  split: 
    select: NoSplit
  
  before_segment_transform:
    - select: PickChannels
      channels: ["C3-M2", "C4-M1", "O1-M2", "O2-M1"]
      source: 'eeg'
      target: 'eeg'
    - select: Concat
      axis: 0
      source: ['eeg', 'eog']
      target: 'ss'
    - select: Concat
      axis: 0
      source: ['chest', 'sao2', 'abd']
      target: 'resp'
    - select: Resample
      desired_freq: 256
      source: 'ss'
      target: 'ss'
    - select: Resample
      desired_freq: 256
      source: 'resp'
      target: 'resp'
    - select: Resample
      desired_freq: 256
      source: 'ecg'
      target: 'ecg'
    - select: Select
      key: ['ss', 'resp', 'ecg']

  offline_signal_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'ss'
      target: 'ss'
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'resp'
      target: 'resp'
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'ecg'
      target: 'ecg'

  offline_label_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'stage'
      target: 'stage'
  online_signal_transform:
    - select: Squeeze
      axis: 0
      source: 'ss'
      target: 'ss'
    - select: Squeeze
      axis: 0
      source: 'resp'
      target: 'resp'
    - select: Squeeze
      axis: 0
      source: 'ecg'
      target: 'ecg'

  online_label_transform:
    - select: Squeeze
      axis: 0
      source: 'stage'
      target: 'stage'
  
model:
  select: sleepfm.sleepfm.SleepFM
  num_classes: 5
  bas_in_channels: 5
  ecg_in_channels: 1
  resp_in_channels: 3
  embedding_dim: 512
  freeze_encoders: true
  pretrained_checkpoint_path: "/home/lingyus/code/PRL/models/sleepfm/checkpoints/best.pt"
  effnet_depth: [1, 2, 2, 3, 3, 3, 3]
  effnet_channels_config: [32, 16, 24, 40, 80, 112, 192, 320, 1280]
  effnet_expansion: 6
  effnet_stride: 2
  effnet_dilation: 1

optimizer:
  lr:  0.001
  select: Adam
  weight_decay: 1e-4

task:
  loss:
    select: CrossEntropyLoss
  select: cinc2018_task.CinC2018Task

trainer:
  fp16: false
  total_epochs: 100
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, precision_macro, f1_macro, recall_macro, roc_auc_macro_ovr, pr_auc_macro,
            precision_weighted, f1_weighted, recall_weighted, roc_auc_weighted_ovr, pr_auc_weighted]
```

------

## 5. 复现步骤

1. **确认配置文件**：确保 `config.yaml` 文件中的数据路径 (`root_path`, `io_path`) 和预训练模型路径 (`model.pretrained_checkpoint_path`) 已修改为你的实际存放路径。

2. **执行实验**：在项目根目录下，运行以下命令。

   ```
   python main.py --config config/cinc2018.yaml
   ```

   *(请将 `config/cinc2018.yaml` 替换为你的实际配置文件路径)*

3. **查看结果**：实验的所有输出将保存在 `common.exp_dir` 指定的目录下，并带有一个时间戳子文件夹。

------

## 6. 预期结果

成功复现本实验后，你应该能获得与下表相似的测试结果。该结果的评估方式为：最佳模型基于训练集训练，通过在验证集上取得的**最佳准确率 (best accuracy)** 进行选择，并最终在测试集上进行评估。

|          | AUROC（Macro） | AUPRC（Macro） | F1 (Macro) |
| -------- | -------------- | -------------- | ---------- |
| Tyee     | 90.27          | 71.08          | 63.9       |
| 官方代码 | 90.14          | 70.35          | 64.7       |

## 附录：数据集划分详情

以下是本实验所使用的训练集、验证集和测试集的受试者名单，与 SleepFM 官方代码保持一致。

### 训练集 (Training Set)

```python
['tr07-0235', 'tr03-0167', 'tr03-0428', 'tr05-0707', 'tr04-0583', 'tr07-0291', 'tr11-0016', 'tr11-0640', 'tr04-1064', 'tr12-0339', 'tr11-0050', 'tr14-0291', 'tr04-1021', 'tr09-0593', 'tr07-0281', 'tr07-0752', 'tr09-0331', 'tr08-0295', 'tr12-0395', 'tr05-0572', 'tr03-0678', 'tr10-0094', 'tr10-0707', 'tr06-0379', 'tr07-0125', 'tr05-0028', 'tr06-0302', 'tr03-0212', 'tr07-0796', 'tr07-0542', 'tr05-1190', 'tr03-0413', 'tr12-0414', 'tr03-0052', 'tr12-0253', 'tr10-0363', 'tr07-0162', 'tr11-0029', 'tr12-0672', 'tr03-0314', 'tr07-0709', 'tr03-1183', 'tr04-0231', 'tr10-0752', 'tr06-0313', 'tr06-0390', 'tr03-0933', 'tr04-0695', 'tr09-0423', 'tr04-0265', 'tr05-1675', 'tr11-0510', 'tr04-0041', 'tr05-1313', 'tr07-0343', 'tr12-0481', 'tr06-0865', 'tr13-0379', 'tr07-0874', 'tr05-0910', 'tr07-0579', 'tr07-0458', 'tr12-0497', 'tr04-0710', 'tr12-0607', 'tr04-1023', 'tr10-0477', 'tr12-0348', 'tr10-0869', 'tr10-0598', 'tr12-0560', 'tr03-0426', 'tr06-0764', 'tr11-0655', 'tr06-0850', 'tr07-0575', 'tr03-1115', 'tr11-0786', 'tr11-0659', 'tr13-0627', 'tr10-0336', 'tr11-0708', 'tr03-0257', 'tr11-0563', 'tr11-0006', 'tr06-0556', 'tr03-0697', 'tr13-0517', 'tr07-0043', 'tr03-0494', 'tr06-0122', 'tr04-0020', 'tr04-0029', 'tr10-0263', 'tr05-1375', 'tr12-0503', 'tr06-0014', 'tr05-1570', 'tr06-0771', 'tr12-0646', 'tr12-0209', 'tr12-0492', 'tr07-0681', 'tr04-1078', 'tr03-0773', 'tr03-0921', 'tr07-0056', 'tr05-0334', 'tr11-0767', 'tr13-0226', 'tr05-1558', 'tr03-0146', 'tr12-0173', 'tr05-0857', 'tr13-0589', 'tr07-0212', 'tr05-1176', 'tr06-0773', 'tr05-1128', 'tr13-0801', 'tr12-0681', 'tr14-0011', 'tr11-0080', 'tr09-0070', 'tr03-0394', 'tr05-0864', 'tr12-0319', 'tr07-0593', 'tr06-0084', 'tr11-0452', 'tr05-1653', 'tr11-0644', 'tr12-0321', 'tr04-0209']

```

### 验证集 (Validation Set)

```python
['tr04-1097', 'tr04-0569', 'tr04-0121', 'tr11-0335', 'tr06-1117', 'tr12-0121', 'tr07-0023', 'tr14-0016', 'tr11-0573', 'tr08-0353', 'tr12-0097', 'tr12-0061', 'tr12-0425', 'tr12-0015', 'tr08-0012']
```

### 测试集 (Test Set)

```python
['tr04-0939', 'tr11-0338', 'tr03-1160', 'tr09-0568', 'tr05-0326', 'tr04-0227', 'tr14-0064', 'tr07-0153', 'tr05-0226', 'tr05-0635', 'tr10-0853', 'tr13-0525', 'tr13-0505', 'tr04-1105', 'tr06-0447', 'tr10-0059', 'tr10-0511', 'tr04-0210', 'tr03-0100', 'tr13-0576', 'tr06-0705', 'tr12-0106', 'tr09-0575', 'tr04-0829', 'tr13-0685', 'tr09-0051', 'tr14-0110', 'tr03-1143', 'tr06-0883', 'tr05-1034', 'tr07-0568', 'tr03-0907', 'tr04-0208', 'tr03-0743', 'tr05-1404', 'tr12-0520', 'tr07-0601', 'tr12-0364', 'tr04-1096', 'tr05-0119', 'tr11-0457', 'tr07-0564', 'tr06-0609', 'tr09-0453', 'tr04-0699', 'tr04-0362', 'tr05-0443', 'tr04-0144', 'tr06-0404', 'tr10-0704', 'tr05-1377', 'tr04-0287', 'tr12-0003', 'tr04-0959', 'tr08-0256', 'tr05-1385', 'tr14-0268', 'tr05-0074', 'tr03-0198', 'tr03-0904', 'tr05-0174', 'tr07-0123', 'tr06-0242', 'tr05-0348', 'tr05-0646', 'tr03-0793', 'tr06-0721', 'tr13-0425', 'tr03-0802', 'tr07-0770', 'tr06-0644', 'tr03-0300', 'tr04-0568', 'tr11-0587', 'tr04-0014', 'tr05-1489', 'tr13-0374', 'tr04-0808', 'tr07-0168', 'tr03-1056', 'tr12-0426', 'tr04-0008', 'tr04-0785', 'tr04-0570', 'tr04-0117', 'tr14-0003', 'tr14-0240', 'tr05-0048', 'tr13-0646', 'tr09-0489', 'tr05-0301', 'tr03-0982', 'tr08-0183', 'tr05-0664', 'tr08-0315', 'tr12-0448', 'tr07-0566', 'tr05-0784', 'tr03-1292', 'tr05-0880']

```