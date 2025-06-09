## Tyee.criterion

`Tyee.criterion` 模块扩展了 PyTorch 现有的损失函数 (`torch.nn.Module`)，提供了在各种深度学习任务中常用的额外自定义损失函数。您可以无缝使用 PyTorch 提供的损失函数，也可以选择 Tyee 中实现的专用损失函数。

以下是 `Tyee.criterion` 中可用的自定义损失函数的摘要：

**自定义损失函数表**

| 类名                         | 功能描述                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| [`LabelSmoothingCrossEntropy`](#1-labelsmoothingcrossentropy) | 实现带有标签平滑的交叉熵损失，用于正则化模型并防止过拟合。   |
| [`SoftTargetCrossEntropy`](#2-softtargetcrossentropy)     | 当目标是软概率分布（而非硬标签）时，计算交叉熵损失。         |
| [`FocalLoss`](#3-focalloss)                  | 实现 Focal Loss，通过降低易分类样本的权重来解决类别不平衡问题。 |
| [`BinnedRegressionLoss`](#4-binnedregressionloss)       | 将连续回归目标转换为分箱概率分布，并计算预测分布与目标分布之间的交叉熵损失。 |

------

### 详细描述

以下是每个自定义损失函数的详细描述，包括其初始化参数和用法。

#### 1. LabelSmoothingCrossEntropy

实现带有标签平滑的交叉熵损失。标签平滑是一种正则化技术，可以防止模型对其预测过于自信。

**初始化:**

```python
LabelSmoothingCrossEntropy(smoothing=0.1)
```

- **`smoothing`** (`float`, 可选): 标签平滑因子。必须小于1.0。一个常用的值是0.1。(默认值: `0.1`)

**用法:**

计算模型预测 (`x`) 与真实目标 (`target`) 之间的损失。

```python
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
loss = criterion(x, target)
```

- **`x`** (`torch.Tensor`): 模型的原始输出 logits (在 softmax 之前)。形状：`(N, C)`，其中 `N` 是批量大小，`C` 是类别数量。
- **`target`** (`torch.Tensor`): 真实标签 (类别索引)。形状：`(N)`。
- **返回** (`torch.Tensor`): 一个标量张量，表示平均损失。

------

#### 2. SoftTargetCrossEntropy

实现针对软目标的交叉熵损失。当真实标签不是硬类别索引而是类别上的概率分布时（例如，来自知识蒸馏或当目标具有固有不确定性时），这非常有用。

**初始化:**

```
SoftTargetCrossEntropy()
```

此损失函数除了 `nn.Module` 的参数外，不需要任何特定的初始化参数。

**用法:**

计算模型预测 (`x`) 与软目标 (`target`) 之间的损失。

```python
criterion = SoftTargetCrossEntropy()
loss = criterion(x, target)
```

- **`x`** (`torch.Tensor`): 模型的原始输出 logits (在 softmax 之前)。形状：`(N, C)`。
- **`target`** (`torch.Tensor`): 真实软标签 (概率分布)。形状：`(N, C)`。每行总和应为1。
- **返回** (`torch.Tensor`): 一个标量张量，表示平均损失。

------

#### 3. FocalLoss

实现 Focal Loss，它对于在类别严重不平衡的数据集上训练模型特别有用。它重塑了标准交叉熵损失，以降低分配给易分类样本的损失权重，从而更关注难以分类的错分样本。

**初始化:**

```python
FocalLoss(gamma=2, alpha=None)
```

- **`gamma`** (`float`, 可选): 聚焦参数。gamma 值越高，对易分类样本的降权作用越强。(默认值: `2`)
- **`alpha`** (`torch.Tensor` 或 `float`, 可选): 每个类别的权重因子。如果是一个 `float`，通常应用于二分类中的正类。如果是一个 `torch.Tensor`，它应该有 `C` 个元素（`C` 是类别数），为每个类别提供一个权重。(默认值: `None`)

**用法:**

计算模型预测 (`logits`) 与真实目标 (`targets`) 之间的 Focal Loss。

```python
# 多分类带 alpha 示例
# alpha_tensor = torch.tensor([0.25, 0.25, 0.25, 0.25]) # 假设有4个类别
# criterion = FocalLoss(gamma=2, alpha=alpha_tensor)

criterion = FocalLoss(gamma=2)
loss = criterion(logits, targets)
```

- **`logits`** (`torch.Tensor`): 模型的原始输出 logits。形状：`(N, C)`。
- **`targets`** (`torch.Tensor`): 真实标签 (类别索引)。形状：`(N)`。
- **返回** (`torch.Tensor`): 一个标量张量，表示平均损失。

------

#### 4. BinnedRegressionLoss

实现一种回归损失函数，其中连续目标变量被离散化到多个分箱中。真实的连续值被转换为这些分箱上的软概率分布（通常使用高斯分布），损失则计算为模型预测分布与此目标分布之间的交叉熵。这常用于估计生理信号（如心率）等任务，在这些任务中预测分布可能更鲁棒。

**初始化:**

```python
BinnedRegressionLoss(dim, min_hz, max_hz, sigma_y)
```

- **`dim`** (`int`): 将目标范围离散化为的分箱数量。
- **`min_hz`** (`float`): 可预测范围的最小值 (单位：赫兹 Hz)。内部会将其转换为每分钟心跳数 (BPM) 以定义第一个分箱的下边界。
- **`max_hz`** (`float`): 可预测范围的最大值 (单位：赫兹 Hz)。内部会将其转换为 BPM 以定义最后一个分箱的上边界。
- **`sigma_y`** (`float`): 用于将连续真实目标值转换为分箱上软分布的高斯分布的标准差 (单位：BPM)。这控制了目标分布的“软度”或扩展程度。

**用法:**

计算分箱回归损失。

```python
criterion = BinnedRegressionLoss(dim=100, min_hz=0.5, max_hz=3.0, sigma_y=2.5)
loss = criterion(y_pred, y_true)
```

- **`y_pred`** (`torch.Tensor`): 模型预测的在分箱上的概率分布。期望是 softmax 层的输出。形状：`(batch_size, seq_len, dim)`。
- **`y_true`** (`torch.Tensor`): 真实的连续目标值 (例如，心率，单位 BPM)。形状：`(batch_size, seq_len)`。
- **返回** (`torch.Tensor`): 一个标量张量，表示平均损失。