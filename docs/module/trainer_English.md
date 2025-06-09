# tyee.trainer

The `tyee.trainer` module is the core training engine of the Tyee framework. It is responsible for taking a defined `Task` (the experiment blueprint) and a configuration file, and then managing the entire training, validation, and evaluation process from start to finish. The `Trainer` encapsulates complex loop logic, distributed training, mixed-precision, logging, and checkpointing, allowing the user to focus on the design of the `Task`.

## Core Functionality of the `Trainer` Class

The `Trainer` class orchestrates and executes the entire experiment through a series of methods.

### 1. Initialization and Component Building (`__init__` and `_init_components`)

- **`__init__(self, cfg, rank, world_size)`**: The constructor for the `Trainer`. It parses the main configuration dictionary `cfg` to set global parameters such as total training steps/epochs, intervals for logging/evaluation/saving, and whether to enable mixed-precision (fp16). Most importantly, it instantiates a `PRLTask` object based on the configuration, which defines the specific behavior of the experiment. The `rank` and `world_size` parameters are used to support distributed training.

- `_init_components(...)`

  : This method is called at the beginning of each training run (e.g., for each fold in cross-validation). It is responsible for preparing all necessary components, including:

  - Creating a logger and TensorBoard `SummaryWriter`.
  - Calling methods on the `Task` object to build and split datasets, and to create `DataLoader`s.
  - Calling methods on the `Task` object to build the model, optimizer, and learning rate scheduler.
  - Setting up Distributed Data Parallel (DDP) and the `GradScaler` for mixed-precision training based on the configuration.
  - Checking for and resuming from a specified checkpoint.

### 2. Main Training Loop (`run` and `run_loop`)

- **`run(self)`**: The public entry point to start the training process. It first gets the dataset splits from the `Task` (supporting cross-validation) and then executes the `run_loop` for each fold.
- **`run_loop(self)`**: This contains the core `for` loop that iterates from a `start_step` to a `total_steps`. It manages the data iterator and calls the `on_...` hook methods defined in the `Task` at the appropriate times (e.g., at the start/end of each epoch).

### 3. Training and Evaluation Steps (`_train_step` and `_eval_step`)

- **`_train_step(...)`**: Executes a **single training step**. It fetches a batch of data from the dataloader, then calls the user-defined `train_step` method from the `Task` to perform the model's forward pass and loss calculation. Afterward, it handles all the underlying operations like backpropagation, gradient accumulation, gradient clipping, optimizer updates, and learning rate adjustments.
- **`_eval_step(...)`**: Executes a **full evaluation round** (on the validation or test set). It iterates through all the data in an evaluation loader, calls the user-defined `valid_step` method from the `Task` for each batch, and uses the `MetricEvaluator` to accumulate and compute the final evaluation metrics.

### 4. Logging and Checkpointing

- **`_log_...` methods**: These methods are responsible for printing and writing training information—such as training loss, learning rate, gradient norm, and validation/test metrics—to the console and TensorBoard at specified intervals.
- **`_save_checkpoint(...)` and `_check_and_save_best(...)`**: These methods handle model saving. The `Trainer` supports saving the latest checkpoint at fixed intervals and also supports saving the best-performing model based on a specified validation metric (e.g., `val_loss` or `val_accuracy`).