# tyee.config

Tyee 使用 YAML 文件作为唯一的配置来源，来管理和驱动整个实验流程。通过修改配置文件，您可以灵活地定义和调整从数据处理、模型选择到训练过程的所有环节，而无需修改任何代码。这种设计使得实验配置清晰、可复现且易于分享。

## 主要配置字段

一个典型的 `config.yaml` 文件由以下几个顶层字段组成，每个字段负责一个特定的模块：

- **`common`**: 通用配置项。
  - 用于设置全局参数，如随机种子 (`seed`)、日志详细程度 (`verbose`) 以及实验结果的输出目录 (`exp_dir`) 等。
- **`dataset`**: 所有与数据相关的配置。
  - **数据来源**: `dataset` (指定要使用的数据集类)、`root_path` (原始数据路径)、`io_path` (预处理后数据的缓存路径)。
  - **加载参数**: `batch_size` (批处理大小)、`num_workers` (数据加载的工作进程数)。
  - **数据划分**: `split` 字段用于定义数据集的划分策略，如 `KFoldPerSubjectCross` (按被试进行K折交叉验证) 或 `NoSplit` (不划分)。
  - **数据变换**: 包含 `offline_..._transform` 和 `online_..._transform` 等字段。这些字段以**列表**形式定义了一系列数据处理流程，允许您像搭积木一样组合各种变换（如滤波、归一化、滑窗等）。
- **`model`**: 定义要使用的模型及其超参数。
  - `select` 字段用于指定模型的类名（例如 `g2g.EncoderNet`）。
  - 该字段下的其他所有键值对都会作为参数传递给模型类的构造函数 `__init__`。
- **`task`**: 定义实验的核心逻辑任务 (`PRLTask` 的子类)。
  - `select` 字段指定要使用的 `Task` 类。
  - `loss` 字段用于配置该任务所使用的损失函数及其参数。
- **`optimizer`**: 配置优化器。
  - `select` 字段指定优化器的名称（如 `AdamW`）。
  - 其他键值对（如 `lr`, `weight_decay`）是传递给优化器的参数。
- **`lr_scheduler`**: 配置学习率调度器。
  - `select` 字段指定调度器的名称（如 `StepLRScheduler`）。
  - 其他键值对是其对应的初始化参数。
- **`trainer`**: 配置训练器，控制整个训练流程。
  - 定义训练的总轮数/步数 (`total_epochs` / `total_steps`)。
  - 设置日志、评估和模型保存的间隔 (`log_interval`, `eval_interval`, `save_interval`)。
  - 指定用于评估和保存最佳模型的指标 (`metrics`, `eval_metric`)。
  - 控制是否启用混合精度训练 (`fp16`)。
  - **断点恢复**: `resume` 子字段用于从之前的训练状态中恢复。将 `enabled` 设为 `true` 以启用，并通过 `checkpoint` 指定要加载的检查点文件路径。
- **`distributed`**: 配置分布式训练。
  - 定义后端 (`backend`) 和进程总数 (`world_size`) 等参数。

## 如何使用

使用 Tyee 框架的核心就是编写和调整 `config.yaml` 文件。

1. **选择与配置组件**: 对于 `model`, `optimizer`, `lr_scheduler`, `loss` 等部分，您只需通过 `select` 字段指定您想使用的类的名称，然后在同一层级下以键值对的形式提供该类的初始化参数即可。框架会自动加载并实例化它们。

2. **定义数据流**: 在 `dataset` 部分，您可以自由组合 `offline` 和 `online` 的数据变换。变换会按照列表中的顺序依次执行，这为您提供了极大的数据处理灵活性。

3. **运行实验**: 准备好您的 `config.yaml` 文件后，通常您会通过一个主训练脚本来启动实验，并将配置文件的路径作为参数传入。例如：

   ```bash
   python train.py --config /path/to/your/experiment_config.yaml
   ```

   `Trainer` 会自动加载该文件，并根据其中的定义来搭建和运行整个实验。