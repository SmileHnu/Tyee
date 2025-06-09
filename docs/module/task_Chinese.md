# tyee.task 

`tyee.task` 模块是**组织和配置**一个完整机器学习实验的核心。它本身不直接执行训练循环，而是扮演一个“实验蓝图”的角色，负责**定义和构建**实验流程中所需的所有关键组件——从数据加载、模型架构到损失函数和优化器。此外，它还定义了单步训练（`train_step`）和验证（`valid_step`）的具体逻辑。一个独立的训练器（Trainer）会调用 Task 实例来执行完整的训练和评估流程。

## `PRLTask` 的核心功能

`PRLTask` 基类提供了一套完整、可重载的方法，用于搭建和管理实验的各个组件。

### 1. 配置驱动 (Configuration-Driven)

- **`__init__(self, cfg)`**: 构造函数接收一个配置字典 `cfg`，并解析其中关于数据集、模型、损失函数、优化器等所有组件的参数设置。这使得整个实验流程可以通过修改配置文件来灵活调整。

### 2. 对象构建 (Object Building)

`PRLTask` 内置了一系列“工厂”方法，用于根据配置动态地创建和实例化 PyTorch 对象：

- **`build_transforms()`**: 根据配置构建数据预处理和增强的变换。
- **`build_dataset()` / `build_datasets()`**: 根据配置构建训练、验证和测试数据集。
- **`build_splitter()`**: 构建数据集划分策略，如 `KFold` 或 `NoSplit`。
- **`build_loss()`**: 根据配置构建损失函数，支持 PyTorch 内置损失和 Tyee 自定义损失。
- **`build_optimizer()`**: 根据配置构建优化器。
- **`build_lr_scheduler()`**: 根据配置构建学习率调度器。

### 3. 数据加载与处理

`PRLTask` 封装了数据加载的通用逻辑：

- **`get_datasets()`**: 整合了数据集构建和划分的完整流程。
- **`build_sampler()` / `build_dataloader()`**: 创建支持分布式训练的采样器和数据加载器。
- **`load_sample()`**: 从数据加载器中获取一个批次的数据，并自动将其移动到指定的计算设备（如 GPU）。

### 4. 训练钩子 (Training Hooks)

`PRLTask` 提供了一系列 `on_...` 方法（如 `on_train_start`, `on_train_epoch_end` 等）。这些方法是空的回调函数（hooks），允许用户在训练流程的特定时间点（如每轮训练开始/结束时）插入自定义逻辑，例如日志记录、模型保存等，而无需修改主训练循环。

## 如何自定义 Task

自定义一个实验任务是使用 Tyee 框架的核心步骤。您只需要继承 `PRLTask` 基类，并实现其中的几个关键方法。

### 第一步：创建子类

首先，创建一个继承自 `PRLTask` 的新类。

```python
from tyee.task import PRLTask

class MyClassificationTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 您可以在这里添加额外的初始化逻辑
```

### 第二步：实现必要方法

`PRLTask` 中有几个方法被定义为必须由子类实现的抽象方法。您需要根据您的具体任务重写它们。

1. **`build_model(self)`** 该方法需要返回一个实例化的 `nn.Module` 模型。

   ```python
   def build_model(self):
       # 从配置中获取模型名称和参数
       model_cls = lazy_import_module(f'model', self.model_select)
       # 实例化模型并返回
       model = model_cls(**self.model_params)
       return model
   ```

2. **`train_step(self, model, sample, ...)`** 该方法定义了**单个训练步骤**的完整逻辑，包括前向传播、损失计算等。它应该返回一个包含必要信息的字典。

   ```python
   def train_step(self, model, sample):
       # 从 sample 中解包数据和标签
       data, labels = sample['signal'], sample['label']
   
       # 前向传播
       outputs = model(data)
   
       # 计算损失
       loss = self.loss(outputs, labels)
   
       # 返回一个包含损失和预测结果的字典，供后续计算指标使用
       return {
           'loss': loss,
           'output': outputs.detach(), # detach() 用于指标计算，避免不必要的梯度信息
           'label': labels.detach()
       }
   ```

3. **`valid_step(self, model, sample, ...)`** 该方法与 `train_step` 类似，但用于**验证步骤**。它通常在 `torch.no_grad()` 上下文中被调用，因此无需计算梯度。

   ```python
   @torch.no_grad()
   def valid_step(self, model, sample):
       # 逻辑通常与 train_step 相似，但不计算梯度
       data, labels = sample['signal'], sample['label']
       outputs = model(data)
       loss = self.loss(outputs, labels)
   
       return {
           'loss': loss,
           'output': outputs,
           'label': labels
       }
   ```

### 第三步：（可选）覆盖其他方法

您可以根据需要覆盖其他方法以实现更高级的定制。

- **`set_optimizer_params(self, ...)`**: 如果您需要实现更复杂的优化策略，比如**分层学习率衰减 (layer-wise learning rate decay)**，您可以重写此方法来为模型的不同部分设置不同的学习率。
- **`on_...` 钩子**: 您可以在这些钩子方法中添加自定义逻辑。