# tyee.task

The `tyee.task` module is the core for **organizing and configuring** a complete machine learning experiment. It does not execute the training loop itself, but rather acts as an "experiment blueprint," responsible for **defining and building** all the key components required in the experimental process—from data loading and model architecture to the loss function and optimizer. Furthermore, it defines the specific logic for a single training step (`train_step`) and validation step (`valid_step`). A separate Trainer then calls the Task instance to execute the full training and evaluation process.

## Core Functionality of `PRLTask`

The `PRLTask` base class provides a complete, overridable set of methods for building and managing the various components of an experiment.

### 1. Configuration-Driven

- **`__init__(self, cfg)`**: The constructor accepts a configuration dictionary `cfg` and parses the parameter settings for all components, including the dataset, model, loss function, optimizer, etc. This allows the entire experimental flow to be flexibly adjusted by modifying the configuration file.

### 2. Object Building

`PRLTask` includes a series of "factory" methods for dynamically creating and instantiating PyTorch objects from the configuration:

- **`build_transforms()`**: Constructs data preprocessing and augmentation transforms from the config.
- **`build_dataset()` / `build_datasets()`**: Constructs the training, validation, and test datasets from the config.
- **`build_splitter()`**: Constructs the dataset splitting strategy, such as `KFold` or `NoSplit`.
- **`build_loss()`**: Constructs the loss function from the config, supporting both built-in PyTorch losses and custom Tyee losses.
- **`build_optimizer()`**: Constructs the optimizer from the config.
- **`build_lr_scheduler()`**: Constructs the learning rate scheduler from the config.

### 3. Data Loading and Handling

`PRLTask` encapsulates the common logic for data loading:

- **`get_datasets()`**: Integrates the entire workflow of dataset building and splitting.
- **`build_sampler()` / `build_dataloader()`**: Creates samplers and dataloaders that support distributed training.
- **`load_sample()`**: Fetches a batch of data from the dataloader and automatically moves it to the specified compute device (e.g., GPU).

### 4. Training Hooks

`PRLTask` provides a series of `on_...` methods (e.g., `on_train_start`, `on_train_epoch_end`). These are empty callback functions (hooks) that allow users to insert custom logic at specific points in the training flow (like at the start/end of each epoch) for tasks such as logging, checkpointing, etc., without modifying the main training loop.

## How to Customize a Task

Customizing an experiment task is a core step in using the Tyee framework. You simply need to inherit from the `PRLTask` base class and implement a few key methods.

### Step 1: Create a Subclass

First, create a new class that inherits from `PRLTask`.

```python
from tyee.task import PRLTask

class MyClassificationTask(PRLTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        # You can add additional initialization logic here
```

### Step 2: Implement Required Methods

Several methods in `PRLTask` are defined as abstract methods that must be implemented by a subclass. You need to override them according to your specific task.

1. **`build_model(self)`** This method should return an instantiated `nn.Module` model.

   ```python
   def build_model(self):
       # Get model name and parameters from the config
       model_cls = lazy_import_module(f'model', self.model_select)
       # Instantiate the model and return it
       model = model_cls(**self.model_params)
       return model
   ```

2. **`train_step(self, model, sample, ...)`** This method defines the complete logic for a **single training step**, including the forward pass, loss calculation, etc. It should return a dictionary containing necessary information.

   ```python
   def train_step(self, model, sample):
       # Unpack data and labels from the sample
       data, labels = sample['signal'], sample['label']
   
       # Forward pass
       outputs = model(data)
   
       # Calculate loss
       loss = self.loss(outputs, labels)
   
       # Return a dictionary with loss and predictions for metric calculation
       return {
           'loss': loss,
           'output': outputs.detach(), # detach() for metric calculation to avoid holding gradients
           'label': labels.detach()
       }
   ```

3. **`valid_step(self, model, sample, ...)`** This method is similar to `train_step` but is used for the **validation step**. It is typically called within a `torch.no_grad()` context, so no gradient calculation is needed.

   ```python
   @torch.no_grad()
   def valid_step(self, model, sample):
       # Logic is often similar to train_step, but without gradient calculation
       data, labels = sample['signal'], sample['label']
       outputs = model(data)
       loss = self.loss(outputs, labels)
   
       return {
           'loss': loss,
           'output': outputs,
           'label': labels
       }
   ```

### Step 3: (Optional) Override Other Methods

You can override other methods as needed for more advanced customization.

- **`set_optimizer_params(self, ...)`**: If you need to implement more complex optimization strategies, such as **layer-wise learning rate decay**, you can override this method to set different learning rates for different parts of the model.
- **`on_...` hooks**: You can add custom logic to these hook methods.