# tyee.config

Tyee uses YAML files as the sole source of configuration to manage and drive the entire experimental process. By modifying a configuration file, you can flexibly define and adjust all aspects of an experiment—from data processing and model selection to the training process—without changing any code. This design makes experiments clear, reproducible, and easy to share.

## Main Configuration Fields

A typical `config.yaml` file is composed of the following top-level fields, each responsible for a specific module:

- **`common`**: General configuration settings.
  - Used to set global parameters such as the random seed (`seed`), logging verbosity (`verbose`), and the output directory for experiments (`exp_dir`).
- **`dataset`**: All configurations related to data.
  - **Data Source**: `dataset` (specifies the dataset class to use), `root_path` (path to raw data), `io_path` (cache path for preprocessed data).
  - **Loading Parameters**: `batch_size`, `num_workers`.
  - **Data Splitting**: The `split` field defines the dataset splitting strategy, such as `KFoldPerSubjectCross` or `NoSplit`.
  - **Data Transformations**: Includes fields like `offline_..._transform` and `online_..._transform`. These fields define a series of data processing steps as a **list**, allowing you to combine various transformations (like filtering, normalization, sliding windows, etc.) like building blocks.
- **`model`**: Defines the model to be used and its hyperparameters.
  - The `select` field specifies the class name of the model (e.g., `g2g.EncoderNet`).
  - All other key-value pairs under this field are passed as arguments to the model's `__init__` constructor.
- **`task`**: Defines the core logic of the experiment (a subclass of `PRLTask`).
  - The `select` field specifies the `Task` class to use.
  - The `loss` field is used to configure the loss function and its parameters for this task.
- **`optimizer`**: Configures the optimizer.
  - The `select` field specifies the name of the optimizer (e.g., `AdamW`).
  - Other key-value pairs (like `lr`, `weight_decay`) are the parameters passed to the optimizer.
- **`lr_scheduler`**: Configures the learning rate scheduler.
  - The `select` field specifies the name of the scheduler (e.g., `StepLRScheduler`).
  - Other key-value pairs are its corresponding initialization parameters.
- **`trainer`**: Configures the `Trainer`, which controls the entire training process.
  - Defines the total number of training epochs/steps (`total_epochs` / `total_steps`).
  - Sets the intervals for logging, evaluation, and model saving (`log_interval`, `eval_interval`, `save_interval`).
  - Specifies the metrics for evaluation and for saving the best model (`metrics`, `eval_metric`).
  - Controls whether to enable mixed-precision training (`fp16`).
  - **Resuming Training**: The `resume` sub-field is used to resume from a previous training state. Set `enabled` to `true` to activate it and specify the path to the checkpoint file with `checkpoint`.
- **`distributed`**: Configures distributed training.
  - Defines parameters like the backend (`backend`) and the total number of processes (`world_size`).

## How to Use

The core of using the Tyee framework is writing and adjusting the `config.yaml` file.

1. **Select and Configure Components**: For sections like `model`, `optimizer`, `lr_scheduler`, and `loss`, you simply specify the name of the class you want to use in the `select` field, and then provide its initialization parameters as key-value pairs at the same level. The framework will automatically load and instantiate them.

2. **Define the Data Flow**: In the `dataset` section, you can freely combine `offline` and `online` data transformations. The transformations will be executed sequentially according to the order in the list, providing you with great flexibility in data processing.

3. **Run an Experiment**: Once your `config.yaml` file is ready, you will typically start the experiment via a main training script, passing the path to your configuration file as an argument. For example:

   ```bash
   python main.py --config /path/to/your/experiment_config.yaml
   ```

   The `Trainer` will automatically load this file and build and run the entire experiment according to the definitions within it.

4. **Command-line Parameter Override**: Tyee supports dynamically overriding any parameter in the configuration file through command-line arguments, without modifying the original YAML file. This is extremely useful for hyperparameter tuning and batch experiments.

   **Basic Syntax**:
   ```bash
   python main.py --config config.yaml --[section] key=value key2=value2
   ```

   **Supported Configuration Sections**:
   - `--trainer`: Override trainer configurations
   - `--dataset`: Override dataset configurations  
   - `--model`: Override model configurations
   - `--task`: Override task configurations
   - `--optimizer`: Override optimizer configurations
   - `--lr_scheduler`: Override learning rate scheduler configurations
   - `--common`: Override common configurations
   - `--distributed`: Override distributed configurations

   **Usage Examples**:

   ```bash
   # Basic parameter override (single level)
   python main.py --config config.yaml --trainer log_interval=10 eval_interval=100
   
   # Nested parameter override (multi-level)  
   python main.py --config config.yaml --trainer eval_metric.select=f1_macro
   
   # Combining multiple configuration sections
   python main.py --config config.yaml \
       --trainer total_epochs=200 log_interval=20 \
       --optimizer lr=0.001 weight_decay=1e-4 \
       --model hidden_size=512 \
       --common seed=42
   
   # Dataset and task configuration override
   python main.py --config config.yaml \
       --dataset batch_size=64 num_workers=8 \
       --task loss.weight_decay=0.01
   
   # Learning rate scheduler parameter override
   python main.py --config config.yaml \
       --lr_scheduler step_size=50 gamma=0.5
   ```

   **Important Notes**:
   - Parameter format must be `key=value` with no spaces around the equals sign
   - For nested fields, use dot notation, e.g., `eval_metric.select=f1_macro`
   - Supports multiple data types: numbers (`lr=0.001`), strings (`device=cuda:0`), booleans (`fp16=true`)
   - Command-line parameters will override corresponding configurations in the YAML file but will not modify the original file