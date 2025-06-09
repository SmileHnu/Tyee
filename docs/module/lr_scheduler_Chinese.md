# tyee.optim.lr_scheduler

`tyee.optim.lr_scheduler` 模块在 PyTorch 内置的学习率调度器基础上进行了扩展。它提供了一个 [`BaseLRScheduler`](#baselrscheduler) 基类，允许用户方便、清晰地创建自定义的学习率调度策略。

**内置学习率调度器**

| 类名                                                         | 功能描述                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`CosineLRScheduler`](#cosinelrscheduler)                   | 实现带热重启和预热的余弦退火（Cosine Annealing）策略。       |
| [`InverseSquareRootScheduler`](#inversesquarerootscheduler) | 根据更新步数的平方根倒数来衰减学习率，并支持预热。           |
| [`ManualScheduler`](#manualscheduler)                       | 根据预定义的轮数（epoch）或步数（step）计划来手动设置学习率。 |
| [`OneCycleScheduler`](#onecyclescheduler)                   | 实现"1Cycle"策略，在一个周期内同时调整学习率和动量。         |
| [`PolynomialDecayLRScheduler`](#polynomialdecaylrscheduler) | 根据多项式函数衰减学习率，并支持预热。                       |
| [`ReduceLROnPlateauScheduler`](#reducelronplateauscheduler) | 当监控的指标停止改善时降低学习率，并支持预热。               |
| [`StepLRScheduler`](#steplrscheduler)                       | 在固定的时间间隔（阶梯式）将学习率乘以一个衰减因子。         |
| [`TriStageLRScheduler`](#tristagelrscheduler)               | 实现三阶段（预热、保持、衰减）的学习率调整策略。             |
| [`TriangularLRScheduler`](#triangularlrscheduler)           | 以三角波形周期性地增加和减少学习率。                         |


## BaseLRScheduler

`BaseLRScheduler` 继承自 PyTorch 的 `LRScheduler`，为所有自定义调度器提供了一个统一的接口和基础。任何希望在 Tyee 框架中使用的自定义调度器都应继承自这个基类。

**初始化参数**

在创建自定义调度器并继承 `BaseLRScheduler` 时，其 `__init__` 方法需要接收以下基本参数：

- **`optimizer`** (`torch.optim.Optimizer`): 被包装的优化器。调度器将修改此优化器中的学习率。
- **`last_step`** (`int`, 可选): 上一个 step 的索引。用于从中断的训练中恢复状态。默认为 `-1`。
- **`metric`** (`str`, 可选): 用于某些调度策略（如 `ReduceLROnPlateau`）的评估指标名称，例如 `'val_loss'`。默认为 `None`。
- **`metric_source`** (`str`, 可选): 指标的来源，用于框架内部识别。默认为 `None`。

**如何创建自定义调度器**

创建自定义调度器非常简单，只需遵循以下两个步骤：

1. **创建新类并继承 `BaseLRScheduler`**。
2. **实现 `get_lr(self)` 方法**。这是核心步骤。您需要在这个方法中定义学习率如何随训练步骤 (`self.last_step`) 变化的逻辑。该方法**必须返回**一个列表，其中包含优化器中每个参数组的新学习率。

[`返回顶部`](#tyeeoptimlr_scheduler)

## CosineLRScheduler

该调度器实现了流行的“带热重启的余弦退火”策略，并增加了可选的线性预热阶段。它非常灵活，适用于多种训练场景。

**调度策略**

`CosineLRScheduler` 的学习率调整分为两个主要阶段：

1. **线性预热 (Warmup)**: 在训练开始的 `warmup_steps` 步内，学习率从一个很小的初始值 (`warmup_start_lr`) 线性增加到优化器设置的基础学习率 (`base_lr`)。这是一个可选阶段。
2. **余弦退火 (Cosine Annealing)**: 预热结束后，学习率会按照余弦函数的形状，从当前的基础学习率平滑地下降到设定的最小学习率 (`min_lr`)。这个下降过程会持续一个周期 (`period_steps`)。
3. 热重启 (Restart): 当一个余弦退火周期结束时，调度器可以“重启”。
   - 学习率会重置，开始一个新的余弦退火周期。
   - 在每次重启时，可以选择性地让下一个周期的时长增加（通过 `t_mult` 参数）和/或让下一个周期的基础学习率降低（通过 `lr_shrink` 参数）。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。用于将以 epoch 为单位的参数转换为以 step 为单位。
- **`period_steps`** (`int`, 可选): 一个余弦退火周期的总步数。
- **`period_epochs`** (`int`, 可选): 一个余弦退火周期的总轮数。如果提供了此参数，它将被转换为 `period_steps`。
- **`warmup_steps`** (`int`, 可选): 线性预热阶段的总步数。
- **`warmup_epochs`** (`int`, 可选): 线性预热阶段的总轮数。如果提供了此参数，它将被转换为 `warmup_steps`。
- **`warmup_start_lr`** (`float`): 预热阶段的起始学习率。默认为 `0.0`。
- **`min_lr`** (`float`): 余弦退火结束后的最小学习率。默认为 `0.0`。
- **`lr_shrink`** (`float`): 每次重启时，基础学习率的缩减因子。例如，`0.5` 表示每次重启后基础学习率减半。`1.0` 表示不缩减。默认为 `1.0`。
- **`t_mult`** (`float`): 每次重启时，周期的增长因子。例如，`2.0` 表示每个新周期的长度是前一个周期的两倍。`1.0` 表示所有周期长度相同。默认为 `1.0`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# 设置优化器的初始学习率为 0.01
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = CosineLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=5,          # 预热 5 个 epoch
    period_epochs=10,         # 第一个余弦周期的长度为 10 个 epoch
    min_lr=1e-5,              # 最小学习率
    t_mult=2,                 # 每次重启后，周期长度乘以 2
    lr_shrink=0.8             # 每次重启后，学习率乘以 0.8
)

# 在训练循环中使用
for epoch in range(40): # 示例总共训练 40 个 epoch
    for step in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        # 更新学习率
        scheduler.step()
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## InverseSquareRootScheduler

该调度器实现了一种学习率衰减策略，学习率会根据更新次数的平方根倒数进行衰减。它也支持一个可选的线性预热阶段。

**调度策略**

`InverseSquareRootScheduler` 的学习率调整主要包括两个阶段：

1. **线性预热 (Warmup)**: 在训练开始的 `warmup_steps` 步内（如果指定了 `warmup_steps` 或 `warmup_epochs`），学习率会从 `warmup_start_lr` 线性增加到优化器中设置的基础学习率 (`base_lr`)。
2. **平方根倒数衰减 (Inverse Square Root Decay)**: 预热阶段结束后，学习率会按照以下公式进行衰减： `lr = decay_factor / sqrt(step)` 其中 `decay_factor` 是基于预热结束时的基础学习率和预热步数计算得到的常量 (`base_lr * sqrt(warmup_steps)`)，`step` 是当前的训练步数。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。用于将以 epoch 为单位的参数转换为以 step 为单位。
- **`warmup_epochs`** (`int`, 可选): 线性预热阶段的总轮数。如果提供了此参数，它将被转换为 `warmup_steps`。
- **`warmup_steps`** (`int`, 可选): 线性预热阶段的总步数。如果两者都未提供，则不进行预热。
- **`warmup_start_lr`** (`float`): 预热阶段的起始学习率。默认为 `0.0`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# 设置优化器的初始学习率为 0.001
optimizer = Adam(model.parameters(), lr=0.001)

# 初始化调度器
scheduler = InverseSquareRootScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=10,             # 预热 10 个 epoch
    warmup_start_lr=1e-7          # 预热起始学习率
)

# 在训练循环中使用
print(f"Base LR: {scheduler.base_lrs[0]}")
for epoch in range(50): # 示例总共训练 50 个 epoch
    for step_in_epoch in range(niter_per_epoch):
        current_global_step = epoch * niter_per_epoch + step_in_epoch
        # ... 训练代码 ...
        # optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        
        # 更新学习率 (传入全局步数)
        # 注意：BaseLRScheduler 的 step 方法默认使用 self.last_step + 1
        # 如果要显式控制步数，可以 scheduler.step(current_global_step)
        # 但通常直接调用 scheduler.step() 即可，它内部会递增 self.last_step
        scheduler.step()
        
        if step_in_epoch == 0 and epoch % 5 == 0:
            print(f"Epoch: {epoch}, Step: {current_global_step}, LR: {scheduler.get_last_lr()[0]:.6e}")
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## ManualScheduler

该调度器允许您根据预定义的计划，在特定的训练轮次 (epoch) 或步骤 (step) 手动设置学习率。它提供了一种实现自定义、分段常数学习率变化的方法。

**调度策略**

`ManualScheduler` 根据您在字典中提供的计划来工作。

1. **定义计划**: 您通过将特定的轮次或步骤映射到期望的学习率来定义一个计划。例如, `{10: 0.001, 20: 0.0001}` 意味着学习率将在第10个 epoch 开始时设置为 `0.001`，然后在第20个 epoch 开始时更改为 `0.0001`。
2. **基于步骤应用**: 调度器以步为单位运行。任何基于轮次的计划都会在内部转换成基于步骤的计划。在任何给定的训练步骤，调度器会查找计划中最近的一个步骤，并应用其对应的学习率。该学习率将保持不变，直到达到下一个计划中的步骤。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。用于将 `epoch2lr` 计划转换为基于步骤的计划。
- **`epoch2lr`** (`dict`, 可选): 一个将 epoch 编号映射到学习率的字典 (例如, `{10: 0.001, 20: 0.0001}`).
- **`step2lr`** (`dict`, 可选): 一个将全局 step 编号映射到学习率的字典。如果同时提供了 `epoch2lr` 和 `step2lr`，它们将被合并。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# 在优化器中设置一个基础学习率，它将在第一次计划变更前被使用
optimizer = Adam(model.parameters(), lr=0.01)

# 定义一个学习率计划
# 在第 5 个 epoch，学习率变为 0.005
# 在第 15 个 epoch，学习率变为 0.001
# 在全局第 2500 步，学习率变为 0.0005
schedule = {
    "epoch2lr": {5: 0.005, 15: 0.001},
    "step2lr": {2500: 0.0005}
}

# 初始化调度器
scheduler = ManualScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    **schedule
)

# 在训练循环中使用
for epoch in range(30):
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()

        # 在特定 epoch 的开始打印学习率以观察变化
        if step_in_epoch == 0 and epoch in [0, 5, 15, 25]:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} 开始: 学习率为 {current_lr}")
# 预期输出:
# Epoch 0 开始: 学习率为 0.01
# Epoch 5 开始: 学习率为 0.005
# Epoch 15 开始: 学习率为 0.001
# Epoch 25 开始: 学习率为 0.0005
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## OneCycleScheduler

该调度器封装了 PyTorch 的 `torch.optim.lr_scheduler.OneCycleLR`.

**调度策略**

1. **预热阶段 (Warm-up)**: 学习率从一个较低的初始值（`max_lr / div_factor`）线性或按余弦曲线增加到设定的最大学习率（`max_lr`）。与此同时，动量从 `max_momentum` 减少到 `base_momentum`。此阶段占总训练步数的 `pct_start` 比例。
2. **冷却阶段 (Cool-down)**: 学习率从 `max_lr` 平滑地下降回初始学习率。动量则反向增加回 `max_momentum`。此阶段占据剩余的训练步数。
3. **湮灭阶段 (Annihilation)**: (可选, 当 `three_phase=True` 时启用) 在训练的最后阶段，学习率会从初始学习率进一步下降到一个非常小的值（`初始学习率 / final_div_factor`），这有助于模型更精细地收敛。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`max_lr`** (`float`): 学习率在周期内能达到的最大学习率。
- **`epochs`** (`int`): 总的训练轮数。`total_steps` 将被计算为 `epochs * niter_per_epoch`。
- **`pct_start`** (`float`): 学习率上升阶段所占的比例。默认为 `0.3`。
- **`anneal_strategy`** (`str`): 学习率的退火策略，可以是 `'cos'` (余弦) 或 `'linear'` (线性)。默认为 `'cos'`。
- **`cycle_momentum`** (`bool`): 如果为 `True`，动量将与学习率反向周期性变化。默认为 `True`。
- **`base_momentum`** (`float`): 动量的下界。默认为 `0.85`。
- **`max_momentum`** (`float`): 动量的上界。默认为 `0.95`。
- **`div_factor`** (`float`): 用于计算初始学习率的除数，`初始学习率 = max_lr / div_factor`。默认为 `25.0`。
- **`final_div_factor`** (`float`): 用于计算最小学习率的除数，`最小学习率 = 初始学习率 / final_div_factor`。默认为 `1e4`。
- **`three_phase`** (`bool`): 如果为 `True`，则启用第三个“湮灭”阶段。默认为 `False`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
# 总训练轮数
epochs = 20

model = torch.nn.Linear(10, 2)
# 注意：优化器中的 lr 实际上会被 OneCycleScheduler 的初始学习率覆盖
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

# 初始化调度器
scheduler = OneCycleScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    epochs=epochs,
    max_lr=0.01,  # 设置最大学习率
    pct_start=0.25, # 25% 的时间用于预热
    div_factor=10   # 初始学习率 = max_lr / 10 = 0.001
)

# 在训练循环中，OneCycleLR 在每个 step 后更新
for epoch in range(epochs):
    for step in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        # 更新学习率和动量
        scheduler.step()

        # 可以在此处获取当前学习率以供监控
        # current_lr = scheduler.get_last_lr()[0]
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## PolynomialDecayLRScheduler

该调度器实现了一种多项式衰减策略。学习率首先可以经历一个可选的线性预热阶段，然后根据一个多项式函数平滑地衰减到最终值。当 `power=1.0` 时，此调度器等效于线性衰减。

**调度策略**

`PolynomialDecayLRScheduler` 的学习率调整分为两个阶段：

1. **线性预热 (Warmup)**: 在训练开始的 `warmup_steps` 步内，学习率从 `warmup_start_lr` 线性增加到优化器设置的基础学习率 (`base_lr`)。
2. **多项式衰减 (Polynomial Decay)**: 预热结束后，学习率会根据一个多项式曲线从基础学习率 (`base_lr`) 衰减到 `end_learning_rate`。衰减过程会持续到训练的总步数 (`total_steps`) 结束。衰减的曲率由 `power` 参数控制。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`total_steps`** (`int`, 可选): 训练的总步数。衰减阶段将持续到此步数。
- **`total_epochs`** (`int`, 可选): 训练的总轮数。如果提供，它将被转换为 `total_steps`。
- **`warmup_epochs`** (`int`, 可选): 线性预热阶段的总轮数。
- **`warmup_steps`** (`int`, 可选): 线性预热阶段的总步数。
- **`warmup_start_lr`** (`float`): 预热阶段的起始学习率。默认为 `0.0`。
- **`end_learning_rate`** (`float`): 衰减结束后的最终学习率。默认为 `0.0`。
- **`power`** (`float`): 多项式衰减的幂次。`1.0` 对应线性衰减。默认为 `1.0`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
import torch
from torch.optim import Adam
from tyee.optim.lr_scheduler import PolynomialDecayLRScheduler # 假设路径正确

# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
total_epochs = 50

model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = PolynomialDecayLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    total_epochs=total_epochs,
    warmup_epochs=5,           # 预热 5 个 epoch
    power=2.0,                 # 使用二次多项式进行衰减
    end_learning_rate=1e-6     # 最终学习率
)

# 在训练循环中使用
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## ReduceLROnPlateauScheduler

该调度器封装了 PyTorch 的 `ReduceLROnPlateau`，并增加了一个可选的线性预热阶段。它可以在一个监控指标（如验证集损失）停止改善时降低学习率。

**调度策略**

`ReduceLROnPlateauScheduler` 的学习率调整分为两个阶段：

1. **线性预热 (Warmup)**: (可选) 在训练开始的 `warmup_steps` 步内，学习率从 `warmup_start_lr` 线性增加到优化器中设置的基础学习率。
2. **平台期检测 (Plateau Detection)**: 预热结束后，调度器会监控一个指定的指标值。如果在 `patience_steps`（耐心步数）内该指标没有明显改善（根据 `mode` 和 `threshold` 判断），学习率将会乘以一个缩减因子 `factor`。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`metric_source`** (`str`): 该参数用于训练框架，以从评估结果中识别并选择正确的指标值传入 `step()` 方法。`metric_source` 指定数据来源（如 `'val'`或`'train'`）。
- **`metric`** (`str`): 该参数用于训练框架，`metric` 指定指标名称（如 `'loss'`）。
- **`patience_epochs`** (`int`): 在降低学习率之前，等待指标无改善的轮数。将被转换为 `patience_steps`。默认为 `10`。
- **`patience_steps`** (`int`, 可选): 在降低学习率之前，等待指标无改善的步数。
- **`factor`** (`float`): 学习率的缩减因子 (`new_lr = lr * factor`)。默认为 `0.1`。
- **`threshold`** (`float`): 用于衡量指标是否有“明显”改善的阈值。默认为 `1e-4`。
- **`mode`** (`str`): `'min'` 或 `'max'` 之一。在 `'min'` 模式下，当指标停止下降时降低学习率；在 `'max'` 模式下，当指标停止上升时降低学习率。默认为 `'min'`。
- **`warmup_epochs`** (`int`, 可选): 线性预热阶段的总轮数。
- **`warmup_steps`** (`int`, 可选): 线性预热阶段的总步数。
- **`warmup_start_lr`** (`float`): 预热阶段的起始学习率。默认为 `0.0`。
- **`min_lr`** (`float`): 学习率可以降低到的下限。默认为 `0.0`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

**重要提示**: `step()` 方法会自动处理预热阶段。在预热阶段之后，您**必须**在调用 `step()` 时传入一个 `metrics` 值来触发平台期检测逻辑。

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = ReduceLROnPlateauScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    metric_source='val',       # 监控验证集指标
    metric='loss',             # 具体监控 'loss' 指标
    mode='min',                # 监控一个需要最小化的指标
    factor=0.5,                # 当指标不改善时，学习率乘以 0.5
    patience_epochs=5,         # 5 个 epoch 内指标不改善则触发
    warmup_epochs=2            # 预热 2 个 epoch
)
# 在训练循环中使用
val_loss = None
for epoch in range(30):
    # --- 训练阶段 ---
    # 在训练的每一步都调用 step()，它会在预热期内正确更新学习率
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step(metrics=val_loss) # 在预热期内，这会更新LR。预热期后，若不带参数调用，则LR不变。

    # --- 验证阶段 ---
    # val_loss = ... 计算验证集损失 ...
    val_loss = 0.5 - epoch * 0.01 # 模拟一个逐渐改善的损失
    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Current LR = {optimizer.param_groups[0]['lr']:.6f}")
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## StepLRScheduler

该调度器实现了经典的阶梯式学习率衰减策略。学习率会在固定的时间间隔（基于轮数或步数）乘以一个衰减因子。它也支持一个可选的线性预热阶段。

**调度策略**

`StepLRScheduler` 的学习率调整分为两个阶段：

1. **线性预热 (Warmup)**: (可选) 在训练开始的 `warmup_steps` 步内，学习率从 `warmup_start_lr` 线性增加到优化器中设置的基础学习率 (`base_lr`)。
2. **阶梯衰减 (Step Decay)**: 预热结束后，每经过 `step_size` 个训练步数，学习率就会乘以一个衰减因子 `gamma`。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`step_size`** (`int`, 可选): 学习率衰减的间隔步数。
- **`epoch_size`** (`int`, 可选): 学习率衰减的间隔轮数。如果提供，它将被转换为 `step_size`。
- **`gamma`** (`float`): 学习率的衰减乘法因子。默认为 `0.1`。
- **`warmup_steps`** (`int`, 可选): 线性预热阶段的总步数。
- **`warmup_epochs`** (`int`, 可选): 线性预热阶段的总轮数。
- **`warmup_start_lr`** (`float`): 预热阶段的起始学习率。默认为 `0.0`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
import torch
from torch.optim import Adam
from tyee.optim.lr_scheduler import StepLRScheduler # 假设路径正确

# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = StepLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    epoch_size=10,             # 每 10 个 epoch 衰减一次学习率
    gamma=0.5,                 # 每次衰减，学习率变为原来的一半
    warmup_epochs=5            # 预热 5 个 epoch
)

# 在训练循环中使用
for epoch in range(30):
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()

    # 在每个 epoch 结束后打印学习率以观察变化
    current_lr = scheduler.get_last_lr()[0]
    print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## TriStageLRScheduler

该调度器实现了一个三阶段学习率调整策略，常用于需要稳定训练峰值性能的场景。它将训练过程分为预热、保持和衰减三个明确的阶段。

**调度策略**

`TriStageLRScheduler` 将学习率的调整分为三个连续的阶段：

1. **预热阶段 (Warmup)**: 在 `warmup_steps` 步内，学习率从一个较低的初始值 (`init_lr`) 线性增加到峰值学习率 (`peak_lr`)。
2. **保持阶段 (Hold)**: 在接下来的 `hold_steps` 步内，学习率保持在 `peak_lr` 不变，允许模型在最高学习率下充分训练。
3. **衰减阶段 (Decay)**: 在最后的 `decay_steps` 步内，学习率从 `peak_lr` 按指数衰减到最终学习率 (`final_lr`)。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`warmup_epochs`** (`int`, 可选): 预热阶段的总轮数。
- **`warmup_steps`** (`int`, 可选): 预热阶段的总步数。
- **`hold_epochs`** (`int`, 可选): 保持阶段的总轮数。
- **`hold_steps`** (`int`, 可选): 保持阶段的总步数。
- **`decay_epochs`** (`int`, 可选): 衰减阶段的总轮数。
- **`decay_steps`** (`int`, 可选): 衰减阶段的总步数。
- **`init_lr_scale`** (`float`): 用于计算初始学习率的缩放因子，`初始学习率 = init_lr_scale * base_lr`。默认为 `0.01`。
- **`final_lr_scale`** (`float`): 用于计算最终学习率的缩放因子，`最终学习率 = final_lr_scale * base_lr`。默认为 `0.01`。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
# 优化器中的 lr 将被用作 peak_lr
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = TriStageLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=10,         # 预热 10 个 epoch
    hold_epochs=20,           # 保持 20 个 epoch
    decay_epochs=20,          # 衰减 20 个 epoch
    init_lr_scale=0.01,
    final_lr_scale=0.05
)

# 在训练循环中使用
total_epochs = 60
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()

    # 在每个阶段的边界打印学习率以观察变化
    if epoch in [0, 9, 10, 29, 30, 49, 59]:
        current_lr = scheduler.get_last_lr()[0]
        print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)

## TriangularLRScheduler

该调度器实现了三角循环学习率策略。在一个周期内，学习率首先从最小值线性增加到最大值，然后在后半周期线性下降回最小值。该调度器支持周期性的重启，并可以在每次重启后调整周期的长度和学习率范围。

**调度策略**

`TriangularLRScheduler` 的学习率在一个或多个周期内呈三角波形变化：

1. **上升阶段**: 在每个周期的前半段，学习率从 `min_lr` 线性增加到 `max_lr`（峰值学习率）。`max_lr` 是从优化器中获取的基础学习率 (`base_lr`)。

2. **下降阶段**: 在每个周期的后半段，学习率从 `max_lr` 线性下降回 `min_lr`。

3. 周期重启 (Cycle Restart)

   : 当一个周期结束后，可以启动一个新的周期。

   - `max_lr` 可以通过乘以 `lr_shrink` 因子而减小。
   - `min_lr` 也可以选择性地进行缩减（如果 `shrink_min=True`）。
   - 下一个周期的长度可以通过乘以 `t_mult` 因子而增长。

**初始化参数**

- **`optimizer`** (`Optimizer`): 被包装的优化器。
- **`niter_per_epoch`** (`int`): 每个 epoch 包含的迭代（step）次数。
- **`period_epochs`** (`int`, 可选): 一个三角周期的总轮数。
- **`period_steps`** (`int`, 可选): 一个三角周期的总步数。
- **`min_lr`** (`float`): 学习率的下界。默认为 `0.0`。
- **`lr_shrink`** (`float`): 每个周期结束后，`max_lr` 的缩减因子。默认为 `1.0` (不缩减)。
- **`shrink_min`** (`bool`): 如果为 `True`，则 `min_lr` 也会在每个周期后按 `lr_shrink` 因子缩减。默认为 `False`。
- **`t_mult`** (`float`): 每个周期结束后，周期长度的增长因子。默认为 `1.0` (长度不变)。
- **`last_step`** (`int`): 上一个 step 的索引。用于恢复训练。默认为 `-1`。

**使用样例**

~~~python
# 假设 niter_per_epoch (每个epoch的步数) 为 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
# 优化器中的 lr 将被用作第一个周期的 max_lr
optimizer = Adam(model.parameters(), lr=0.01)

# 初始化调度器
scheduler = TriangularLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    period_epochs=10,         # 每个周期为 10 个 epoch
    min_lr=1e-4,              # 学习率最低为 1e-4
    lr_shrink=0.9,            # 每个周期后，max_lr 变为原来的 90%
    t_mult=1.5                # 每个周期后，长度变为原来的 1.5 倍
)

# 在训练循环中使用
total_epochs = 40
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`返回顶部`](#tyeeoptimlr_scheduler)