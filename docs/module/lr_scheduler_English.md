# tyee.optim.lr_scheduler

The `tyee.optim.lr_scheduler` module extends PyTorch's built-in learning rate schedulers. It provides a [`BaseLRScheduler`](#baselrscheduler) class, allowing users to easily and clearly create custom learning rate scheduling strategies.

**Built-in Learning Rate Schedulers**

| Class Name                                                   | Functional Description                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`CosineLRScheduler`](#cosinelrscheduler)                   | Implements the Cosine Annealing strategy with warm restarts and warmup. |
| [`InverseSquareRootScheduler`](#inversesquarerootscheduler) | Decays the learning rate based on the inverse square root of the update step, with warmup. |
| [`ManualScheduler`](#manualscheduler)                       | Manually sets the learning rate at predefined epochs or steps. |
| [`OneCycleScheduler`](#onecyclescheduler)                   | Implements the "1Cycle" policy, adjusting both LR and momentum in a cycle. |
| [`PolynomialDecayLRScheduler`](#polynomialdecaylrscheduler) | Decays the learning rate according to a polynomial function, with warmup. |
| [`ReduceLROnPlateauScheduler`](#reducelronplateauscheduler) | Reduces the learning rate when a monitored metric stops improving, with warmup. |
| [`StepLRScheduler`](#steplrscheduler)                       | Decays the learning rate by a multiplicative factor at fixed intervals (steps). |
| [`TriStageLRScheduler`](#tristagelrscheduler)               | Implements a three-stage (warmup, hold, decay) learning rate strategy. |
| [`TriangularLRScheduler`](#triangularlrscheduler)           | Cyclically increases and decreases the learning rate in a triangular pattern. |


## BaseLRScheduler

`BaseLRScheduler` inherits from PyTorch's `LRScheduler`, providing a unified interface and foundation for all custom schedulers. Any custom scheduler intended for use within the Tyee framework should inherit from this base class.

**Initialization Parameters**

When creating a custom scheduler and inheriting from `BaseLRScheduler`, its `__init__` method needs to accept the following base parameters:

- **`optimizer`** (`torch.optim.Optimizer`): The wrapped optimizer whose learning rate will be modified by the scheduler.
- **`last_step`** (`int`, optional): The index of the last step. Used to resume a training run from an interruption. Defaults to `-1`.
- **`metric`** (`str`, optional): The name of the metric to be used for certain scheduling strategies (like `ReduceLROnPlateau`), e.g., `'loss'`. Defaults to `None`.
- **`metric_source`** (`str`, optional): The source of the metric, used for internal framework identification. Defaults to `None`.

**How to Create a Custom Scheduler**

Creating a custom scheduler is straightforward and involves just two steps:

1. **Create a new class that inherits from `BaseLRScheduler`**.
2. **Implement the `get_lr(self)` method**. This is the core step. In this method, you must define the logic for how the learning rate should change with the training step (`self.last_step`). This method **must return** a list containing the new learning rate for each parameter group in the optimizer.

[`Back to Top`](#tyeeoptimlr_scheduler)

## CosineLRScheduler

This scheduler implements the popular "cosine annealing with warm restarts" strategy, augmented with an optional linear warmup phase. It is highly flexible and suitable for a variety of training scenarios.

**Scheduling Strategy**

The learning rate adjustment in `CosineLRScheduler` is divided into two main phases:

1. **Linear Warmup**: For the first `warmup_steps`, the learning rate linearly increases from a small initial value (`warmup_start_lr`) to the base learning rate (`base_lr`) set in the optimizer. This is an optional phase.
2. **Cosine Annealing**: After warmup, the learning rate smoothly decreases from the current base learning rate to a specified minimum learning rate (`min_lr`), following the shape of a cosine curve. This decay process lasts for one period (`period_steps`).
3. **Warm Restart**: When a cosine annealing period ends, the scheduler can "restart."
   - The learning rate is reset, initiating a new cosine annealing cycle.
   - At each restart, you can optionally increase the duration of the next period (via the `t_mult` parameter) and/or decrease the base learning rate for the next period (via the `lr_shrink` parameter).

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch. Used to convert epoch-based parameters to step-based ones.
- **`period_steps`** (`int`, optional): The total number of steps for one cosine annealing period.
- **`period_epochs`** (`int`, optional): The total number of epochs for one cosine annealing period. If provided, it will be converted to `period_steps`.
- **`warmup_steps`** (`int`, optional): The total number of steps for the linear warmup phase.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the linear warmup phase. If provided, it will be converted to `warmup_steps`.
- **`warmup_start_lr`** (`float`): The starting learning rate for the warmup phase. Defaults to `0.0`.
- **`min_lr`** (`float`): The minimum learning rate after cosine annealing. Defaults to `0.0`.
- **`lr_shrink`** (`float`): The factor by which to shrink the base learning rate at each restart. For example, `0.5` halves the base LR after each restart. `1.0` means no shrinkage. Defaults to `1.0`.
- **`t_mult`** (`float`): The factor by which to grow the period length at each restart. For example, `2.0` means each new period is twice as long as the previous one. `1.0` means all periods have the same length. Defaults to `1.0`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# Set the initial learning rate in the optimizer to 0.01
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = CosineLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=5,          # Warm up for 5 epochs
    period_epochs=10,         # The first cosine period is 10 epochs long
    min_lr=1e-5,              # Minimum learning rate
    t_mult=2,                 # Multiply period length by 2 after each restart
    lr_shrink=0.8             # Multiply learning rate by 0.8 after each restart
)

# Use in the training loop
for epoch in range(40): # Example: train for a total of 40 epochs
    for step in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        # Update the learning rate
        scheduler.step()

~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## InverseSquareRootScheduler

This scheduler implements a learning rate decay strategy where the learning rate decays according to the inverse square root of the update number. It also supports an optional linear warmup phase. 

**Scheduling Strategy**

The learning rate adjustment in `InverseSquareRootScheduler` primarily involves two phases:

1. **Linear Warmup**: For the first `warmup_steps` (if `warmup_steps` or `warmup_epochs` is specified), the learning rate linearly increases from `warmup_start_lr` to the base learning rate (`base_lr`) set in the optimizer.
2. **Inverse Square Root Decay**: After the warmup phase, the learning rate decays according to the formula: `lr = decay_factor / sqrt(step)` where `decay_factor` is a constant calculated based on the base learning rate at the end of warmup and the number of warmup steps (`base_lr * sqrt(warmup_steps)`), and `step` is the current training step.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch. Used to convert epoch-based parameters to step-based ones.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the linear warmup phase. If provided, it will be converted to `warmup_steps`.
- **`warmup_steps`** (`int`, optional): The total number of steps for the linear warmup phase. If neither is provided, no warmup is performed.
- **`warmup_start_lr`** (`float`): The starting learning rate for the warmup phase. Defaults to `0.0`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# Set the initial learning rate in the optimizer to 0.001
optimizer = Adam(model.parameters(), lr=0.001)

# Initialize the scheduler
scheduler = InverseSquareRootScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=10,             # Warm up for 10 epochs
    warmup_start_lr=1e-7          # Warmup starting learning rate
)

# Use in the training loop
print(f"Base LR: {scheduler.base_lrs[0]}")
for epoch in range(50): # Example: train for a total of 50 epochs
    for step_in_epoch in range(niter_per_epoch):
        current_global_step = epoch * niter_per_epoch + step_in_epoch
        # ... training code ...
        # optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        
        # Update the learning rate
        # Note: The step method of BaseLRScheduler uses self.last_step + 1 by default.
        # If explicit step control is needed, you can use scheduler.step(current_global_step).
        # However, usually just calling scheduler.step() is sufficient as it increments self.last_step internally.
        scheduler.step()

        if step_in_epoch == 0 and epoch % 5 == 0:
            print(f"Epoch: {epoch}, Step: {current_global_step}, LR: {scheduler.get_last_lr()[0]:.6e}")
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## ManualScheduler

This scheduler allows you to manually set the learning rate at specific training epochs or steps according to a predefined schedule. It provides a way to implement custom, piecewise-constant learning rate changes.

**Scheduling Strategy**

The `ManualScheduler` works based on a schedule you provide in a dictionary.

1. **Schedule Definition**: You define a schedule by mapping specific epochs or steps to desired learning rates. For example, `{10: 0.001, 20: 0.0001}` means the learning rate will be set to `0.001` at the beginning of epoch 10 and then changed to `0.0001` at the beginning of epoch 20.
2. **Step-based Application**: The scheduler operates on a step-by-step basis. Any epoch-based schedules are internally converted to step-based schedules. At any given training step, the scheduler finds the most recent step defined in the schedule and applies its corresponding learning rate. The learning rate remains constant until the next scheduled step is reached.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch. This is used to convert the `epoch2lr` schedule to a step-based schedule.
- **`epoch2lr`** (`dict`, optional): A dictionary mapping epoch numbers to learning rates (e.g., `{10: 0.001, 20: 0.0001}`).
- **`step2lr`** (`dict`, optional): A dictionary mapping global step numbers to learning rates. If both `epoch2lr` and `step2lr` are provided, they will be merged.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100

model = torch.nn.Linear(10, 2)
# Set a base learning rate in the optimizer. This will be used until the first scheduled change.
optimizer = Adam(model.parameters(), lr=0.01)

# Define a learning rate schedule
# At epoch 5, LR becomes 0.005
# At epoch 15, LR becomes 0.001
# At global step 2500, LR becomes 0.0005
schedule = {
    "epoch2lr": {5: 0.005, 15: 0.001},
    "step2lr": {2500: 0.0005}
}

# Initialize the scheduler
scheduler = ManualScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    **schedule
)

# Use in the training loop
for epoch in range(30):
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step()

        # Print LR at the beginning of certain epochs to observe changes
        if step_in_epoch == 0 and epoch in [0, 5, 15, 25]:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Start of Epoch {epoch}: LR is {current_lr}")
# Expected output:
# Start of Epoch 0: LR is 0.01
# Start of Epoch 5: LR is 0.005
# Start of Epoch 15: LR is 0.001
# Start of Epoch 25: LR is 0.0005

~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## OneCycleScheduler

This scheduler wraps PyTorch's `torch.optim.lr_scheduler.OneCycleLR`.

**Scheduling Strategy**

1. **Warm-up Phase**: The learning rate starts at a low initial value (`max_lr / div_factor`) and increases, either linearly or with a cosine curve, to a specified maximum learning rate (`max_lr`). Concurrently, momentum decreases from `max_momentum` to `base_momentum`. This phase takes up a fraction of the total training steps, defined by `pct_start`.
2. **Cool-down Phase**: The learning rate smoothly decreases from `max_lr` back to the initial learning rate. Momentum increases in reverse, back to `max_momentum`. This phase occupies the remaining training steps.
3. **Annihilation Phase**: (Optional, enabled when `three_phase=True`) In the final stage of training, the learning rate further decays from the initial learning rate to a very small value (`initial_lr / final_div_factor`), which can help the model converge more finely.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`max_lr`** (`float`): The maximum learning rate reached during the cycle.
- **`epochs`** (`int`): The total number of training epochs. `total_steps` will be calculated as `epochs * niter_per_epoch`.
- **`pct_start`** (`float`): The percentage of the cycle spent on increasing the learning rate. Defaults to `0.3`.
- **`anneal_strategy`** (`str`): The annealing strategy for the learning rate, can be `'cos'` (cosine) or `'linear'`. Defaults to `'cos'`.
- **`cycle_momentum`** (`bool`): If `True`, momentum is cycled in reverse to the learning rate. Defaults to `True`.
- **`base_momentum`** (`float`): The lower bound for momentum. Defaults to `0.85`.
- **`max_momentum`** (`float`): The upper bound for momentum. Defaults to `0.95`.
- **`div_factor`** (`float`): Determines the initial learning rate by `initial_lr = max_lr / div_factor`. Defaults to `25.0`.
- **`final_div_factor`** (`float`): Determines the minimum learning rate by `min_lr = initial_lr / final_div_factor`. Defaults to `1e4`.
- **`three_phase`** (`bool`): If `True`, enables the third "annihilation" phase. Defaults to `False`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
import torch
from torch.optim import SGD
from tyee.optim.lr_scheduler import OneCycleScheduler # Assuming correct import path

# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
# Total number of training epochs
epochs = 20

model = torch.nn.Linear(10, 2)
# Note: The lr in the optimizer will be overridden by the initial LR of the OneCycleScheduler.
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

# Initialize the scheduler
scheduler = OneCycleScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    epochs=epochs,
    max_lr=0.01,       # Set the maximum learning rate
    pct_start=0.25,    # Use 25% of the steps for warmup
    div_factor=10      # Initial LR = max_lr / 10 = 0.001
)

# In the training loop, OneCycleLR is updated after each step
for epoch in range(epochs):
    for step in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        # Update learning rate and momentum
        scheduler.step()
        
        # You can get the current LR here for monitoring
        # current_lr = scheduler.get_last_lr()[0]
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## PolynomialDecayLRScheduler

This scheduler implements a polynomial decay strategy. The learning rate can first undergo an optional linear warmup phase, after which it smoothly decays to a final value according to a polynomial function. When `power=1.0`, this scheduler is equivalent to linear decay.

**Scheduling Strategy**

The learning rate adjustment in `PolynomialDecayLRScheduler` is divided into two phases:

1. **Linear Warmup**: For the first `warmup_steps`, the learning rate linearly increases from `warmup_start_lr` to the base learning rate (`base_lr`) set in the optimizer.
2. **Polynomial Decay**: After warmup, the learning rate decays from the base learning rate (`base_lr`) to `end_learning_rate` following a polynomial curve. The decay process lasts until the total number of training steps (`total_steps`) is completed. The curvature of the decay is controlled by the `power` parameter.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`total_steps`** (`int`, optional): The total number of training steps. The decay phase will last until this step.
- **`total_epochs`** (`int`, optional): The total number of training epochs. If provided, it will be converted to `total_steps`.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the linear warmup phase.
- **`warmup_steps`** (`int`, optional): The total number of steps for the linear warmup phase.
- **`warmup_start_lr`** (`float`): The starting learning rate for the warmup phase. Defaults to `0.0`.
- **`end_learning_rate`** (`float`): The final learning rate after decay. Defaults to `0.0`.
- **`power`** (`float`): The power of the polynomial decay. `1.0` corresponds to linear decay. Defaults to `1.0`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
import torch
from torch.optim import Adam
from tyee.optim.lr_scheduler import PolynomialDecayLRScheduler # Assuming correct import path

# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
total_epochs = 50

model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = PolynomialDecayLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    total_epochs=total_epochs,
    warmup_epochs=5,           # Warm up for 5 epochs
    power=2.0,                 # Use quadratic polynomial for decay
    end_learning_rate=1e-6     # Final learning rate
)

# Use in the training loop
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step()
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## ReduceLROnPlateauScheduler

This scheduler wraps PyTorch's `ReduceLROnPlateau` and adds an optional linear warmup phase. It reduces the learning rate when a monitored metric (such as validation loss) has stopped improving.

**Scheduling Strategy**

The learning rate adjustment in `ReduceLROnPlateauScheduler` is divided into two phases:

1. **Linear Warmup** (Optional): For the first `warmup_steps`, the learning rate linearly increases from `warmup_start_lr` to the base learning rate set in the optimizer.
2. **Plateau Detection**: After warmup, the scheduler monitors a specified metric value. If the metric shows no significant improvement (judged by `mode` and `threshold`) for a number of steps (`patience_steps`), the learning rate is reduced by multiplying it by a `factor`.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`metric_source`** (`str`): The parameter is used by the trainer to identify and select the correct metric value from the evaluation results to pass to the `step()` method. `metric_source` specifies the data source (e.g., `'val'`, `'train'`).
- **`metric`** (`str`): The parameter is used by the trainer to identify and select the correct metric value from the evaluation results to pass to the `step()` method.`metric` specifies the metric name (e.g., `'loss'`).
- **`patience_epochs`** (`int`): The number of epochs with no improvement after which the learning rate will be reduced. This is converted to `patience_steps`. Defaults to `10`.
- **`patience_steps`** (`int`, optional): The number of steps with no improvement after which the learning rate will be reduced.
- **`factor`** (`float`): The factor by which the learning rate will be reduced (`new_lr = lr * factor`). Defaults to `0.1`.
- **`threshold`** (`float`): The threshold for measuring if a change is significant. Defaults to `1e-4`.
- **`mode`** (`str`): One of `'min'` or `'max'`. In `'min'` mode, the LR is reduced when the metric stops decreasing. In `'max'` mode, it is reduced when the metric stops increasing. Defaults to `'min'`.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the linear warmup phase.
- **`warmup_steps`** (`int`, optional): The total number of steps for the linear warmup phase.
- **`warmup_start_lr`** (`float`): The starting learning rate for the warmup phase. Defaults to `0.0`.
- **`min_lr`** (`float`): The lower bound to which the learning rate can be reduced. Defaults to `0.0`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

**Important Note**: The `step()` method automatically handles the warmup phase. After warmup, you **must** pass a `metrics` value when calling `step()` to trigger the plateau detection logic. This is typically done after each epoch's validation phase.

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = ReduceLROnPlateauScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    metric_source='val',       # Monitor validation set metrics
    metric='loss',             # Specifically monitor the 'loss' metric
    mode='min',                # Monitor a metric that needs to be minimized
    factor=0.5,                # Reduce LR by a factor of 0.5 when plateauing
    patience_epochs=5,         # Trigger after 5 epochs of no improvement
    warmup_epochs=2            # Warm up for 2 epochs
)

# Use in the training loop
val_loss = None
for epoch in range(30):
    # --- Training Phase ---
    # Call step() after each training step. It will correctly update the LR during warmup.
    # After warmup, if called without arguments, it will keep the LR constant.
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step(metrics=val_loss)

    # --- Validation Phase ---
    # val_loss = ... calculate validation loss ...
    val_loss = 0.5 - epoch * 0.01 # Simulate an improving loss
    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Current LR = {optimizer.param_groups[0]['lr']:.6f}")

~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## StepLRScheduler

This scheduler implements the classic step learning rate decay strategy. The learning rate is multiplied by a decay factor at fixed intervals (based on epochs or steps). It also supports an optional linear warmup phase.

**Scheduling Strategy**

The learning rate adjustment in `StepLRScheduler` is divided into two phases:

1. **Linear Warmup** (Optional): For the first `warmup_steps`, the learning rate linearly increases from `warmup_start_lr` to the base learning rate (`base_lr`) set in the optimizer.
2. **Step Decay**: After the warmup phase, the learning rate is multiplied by a decay factor `gamma` every `step_size` training steps.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`step_size`** (`int`, optional): The number of steps between each learning rate decay.
- **`epoch_size`** (`int`, optional): The number of epochs between each learning rate decay. If provided, it will be converted to `step_size`.
- **`gamma`** (`float`): The multiplicative factor for learning rate decay. Defaults to `0.1`.
- **`warmup_steps`** (`int`, optional): The total number of steps for the linear warmup phase.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the linear warmup phase.
- **`warmup_start_lr`** (`float`): The starting learning rate for the warmup phase. Defaults to `0.0`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = StepLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    epoch_size=10,             # Decay LR every 10 epochs
    gamma=0.5,                 # Halve the LR at each decay step
    warmup_epochs=5            # Warm up for 5 epochs
)

# Use in the training loop
for epoch in range(30):
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step()

    # Print LR at the end of each epoch to observe changes
    current_lr = scheduler.get_last_lr()[0]
    print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## TriStageLRScheduler

This scheduler implements a three-stage learning rate adjustment strategy, often used in scenarios that require stable training at a peak performance level. It divides the training process into three distinct phases: warmup, hold, and decay.

**Scheduling Strategy**

`TriStageLRScheduler` divides the learning rate adjustment into three consecutive phases:

1. **Warmup Phase**: For the first `warmup_steps`, the learning rate linearly increases from a low initial value (`init_lr`) to a peak learning rate (`peak_lr`).
2. **Hold Phase**: For the next `hold_steps`, the learning rate is held constant at `peak_lr`, allowing the model to train sufficiently at the maximum learning rate.
3. **Decay Phase**: For the final `decay_steps`, the learning rate decays exponentially from `peak_lr` to a final learning rate (`final_lr`).

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`warmup_epochs`** (`int`, optional): The total number of epochs for the warmup phase.
- **`warmup_steps`** (`int`, optional): The total number of steps for the warmup phase.
- **`hold_epochs`** (`int`, optional): The total number of epochs for the hold phase.
- **`hold_steps`** (`int`, optional): The total number of steps for the hold phase.
- **`decay_epochs`** (`int`, optional): The total number of epochs for the decay phase.
- **`decay_steps`** (`int`, optional): The total number of steps for the decay phase.
- **`init_lr_scale`** (`float`): A scaling factor to compute the initial learning rate: `initial_lr = init_lr_scale * base_lr`. Defaults to `0.01`.
- **`final_lr_scale`** (`float`): A scaling factor to compute the final learning rate: `final_lr = final_lr_scale * base_lr`. Defaults to `0.01`.
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
# The lr in the optimizer will be used as the peak_lr
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = TriStageLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    warmup_epochs=10,         # Warm up for 10 epochs
    hold_epochs=20,           # Hold for 20 epochs
    decay_epochs=20,          # Decay for 20 epochs
    init_lr_scale=0.01,
    final_lr_scale=0.05
)

# Use in the training loop
total_epochs = 60
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step()

    # Print LR at the boundaries of each stage to observe changes
    if epoch in [0, 9, 10, 29, 30, 49, 59]:
        current_lr = scheduler.get_last_lr()[0]
        print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)

## TriangularLRScheduler

This scheduler implements a triangular cyclical learning rate policy. Within a cycle, the learning rate first increases linearly from a minimum to a maximum value and then decreases linearly back to the minimum in the second half. The scheduler supports periodic restarts and can adjust the cycle length and learning rate range after each restart.

**Scheduling Strategy**

The learning rate in `TriangularLRScheduler` follows a triangular wave pattern over one or more cycles:

1. **Ramp-up Phase**: In the first half of each cycle, the learning rate linearly increases from `min_lr` to `max_lr` (the peak learning rate). The `max_lr` is taken from the base learning rate (`base_lr`) in the optimizer.

2. **Ramp-down Phase**: In the second half of each cycle, the learning rate linearly decreases from `max_lr` back to `min_lr`.

3. Cycle Restart

   : When one cycle ends, a new one can begin.

   - `max_lr` can be reduced by multiplying it by the `lr_shrink` factor.
   - `min_lr` can also be optionally shrunk (if `shrink_min=True`).
   - The length of the next cycle can be increased by multiplying it by the `t_mult` factor.

**Initialization Parameters**

- **`optimizer`** (`Optimizer`): The wrapped optimizer.
- **`niter_per_epoch`** (`int`): The number of iterations (steps) per epoch.
- **`period_epochs`** (`int`, optional): The total number of epochs for one triangular cycle.
- **`period_steps`** (`int`, optional): The total number of steps for one triangular cycle.
- **`min_lr`** (`float`): The lower bound for the learning rate. Defaults to `0.0`.
- **`lr_shrink`** (`float`): The factor by which `max_lr` is shrunk after each cycle. Defaults to `1.0` (no shrinkage).
- **`shrink_min`** (`bool`): If `True`, `min_lr` is also shrunk by the `lr_shrink` factor after each cycle. Defaults to `False`.
- **`t_mult`** (`float`): The factor by which the period length is multiplied after each cycle. Defaults to `1.0` (constant length).
- **`last_step`** (`int`): The index of the last step. Used for resuming training. Defaults to `-1`.

**Usage Example**

~~~python
# Assume niter_per_epoch (steps per epoch) is 100
niter_per_epoch = 100
model = torch.nn.Linear(10, 2)
# The lr in the optimizer will be used as the max_lr for the first cycle
optimizer = Adam(model.parameters(), lr=0.01)

# Initialize the scheduler
scheduler = TriangularLRScheduler(
    optimizer,
    niter_per_epoch=niter_per_epoch,
    period_epochs=10,         # Each cycle is 10 epochs long
    min_lr=1e-4,              # Minimum learning rate is 1e-4
    lr_shrink=0.9,            # After each cycle, max_lr becomes 90% of its previous value
    t_mult=1.5                # After each cycle, its length becomes 1.5x longer
)

# Use in the training loop
total_epochs = 40
for epoch in range(total_epochs):
    for step_in_epoch in range(niter_per_epoch):
        # ... training code ...
        optimizer.step()
        scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"End of Epoch {epoch}: LR is {current_lr:.6f}")
~~~

[`Back to Top`](#tyeeoptimlr_scheduler)