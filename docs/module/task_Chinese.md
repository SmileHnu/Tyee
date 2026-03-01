# tyee.tasks

`tyee.tasks` 的统一任务入口是 `BaseTask`。当前推荐优先使用**配置驱动**方式，不再要求每个实验都单独实现一个 `xxx_task` 子类。

## 核心变化（当前推荐）

1. **统一选择器**：`task.select: base_task.BaseTask`
2. **统一输入映射**：`task.model.input_map`
3. **统一标签映射**：`task.target_map`

`BaseTask.train_step()` / `BaseTask.valid_step()` 会依据上述映射自动：
- 从 `sample` 中取模型输入；
- 前向推理；
- 取目标并计算损失；
- 返回 `{loss, output, label}`。

## 配置写法

### 常见写法（列表映射）

```yaml
task:
    loss:
        select: CrossEntropyLoss
    select: base_task.BaseTask
    model:
        input_map: ['eeg']
    target_map: ['event']
```

### 多输入模型

```yaml
task:
    loss:
        select: CrossEntropyLoss
    select: base_task.BaseTask
    model:
        input_map: ['eeg', 'eog']
    target_map: ['stage']
```

### 字典映射（按关键字传参）

```yaml
task:
    loss:
        select: MSELoss
    select: base_task.BaseTask
    model:
        input_map:
            x: eeg
            mask: eeg_mask
    target_map:
        y: label
```

## BaseTask 提供的能力

- `build_transforms()`：构建离线/在线 transform。
- `build_dataset()` / `build_datasets()`：构建 train/val/test 数据集。
- `build_splitter()`：根据 `dataset.split` 构建切分器。
- `build_model()`：按 `model.select` 动态实例化模型。
- `build_loss()` / `build_optimizer()` / `build_lr_scheduler()`：构建训练组件。
- `get_datasets()`：执行数据构建 + 切分流程。

## 何时仍需自定义任务子类

如需以下高级逻辑，可继承 `BaseTask` 并覆盖方法：

- 特殊前向/后处理流程（非通用映射可表达）；
- 自定义损失输入组织；
- 分层学习率等参数分组（覆盖 `set_optimizer_params()`）；
- 训练生命周期钩子（`on_train_start` 等）。

示例：

```python
from tyee.tasks import BaseTask

class MyTask(BaseTask):
        def train_step(self, model, sample, *args, **kwargs):
                return super().train_step(model, sample, *args, **kwargs)
```