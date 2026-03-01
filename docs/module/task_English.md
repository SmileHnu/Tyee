# tyee.tasks

The unified task entry in `tyee.tasks` is `BaseTask`. The recommended workflow is now **configuration-driven**, so most experiments no longer need a dedicated `xxx_task` subclass.

## Current Recommended Interface

1. Unified selector: `task.select: base_task.BaseTask`
2. Unified model input mapping: `task.model.input_map`
3. Unified supervision mapping: `task.target_map`

`BaseTask.train_step()` / `BaseTask.valid_step()` use these mappings to automatically:
- collect model inputs from `sample`;
- run forward pass;
- collect targets and compute loss;
- return `{loss, output, label}`.

## YAML Patterns

### Common single-input case

```yaml
task:
    loss:
        select: CrossEntropyLoss
    select: base_task.BaseTask
    model:
        input_map: ['eeg']
    target_map: ['event']
```

### Multi-input model

```yaml
task:
    loss:
        select: CrossEntropyLoss
    select: base_task.BaseTask
    model:
        input_map: ['eeg', 'eog']
    target_map: ['stage']
```

### Keyword-argument mapping

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

## What `BaseTask` Provides

- `build_transforms()`
- `build_dataset()` / `build_datasets()`
- `build_splitter()`
- `build_model()`
- `build_loss()` / `build_optimizer()` / `build_lr_scheduler()`
- `get_datasets()` (dataset construction + splitting pipeline)

## When to Still Implement a Custom Task Subclass

Inherit from `BaseTask` only when you need advanced behavior, such as:

- custom forward/post-processing not expressible by maps;
- custom loss input organization;
- advanced optimizer grouping (`set_optimizer_params()`);
- lifecycle hooks (`on_train_start`, etc.).

Example:

```python
from tyee.tasks import BaseTask

class MyTask(BaseTask):
        def train_step(self, model, sample, *args, **kwargs):
                return super().train_step(model, sample, *args, **kwargs)
```