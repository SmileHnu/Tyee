# DEAP - Emotion Recognition

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | DEAP (Database for Emotion Analysis using Physiological Signals) |
| **Signal Type**     | GSR, Respiration, PPG, Temperature                           |
| **Analysis Task**   | Emotion Recognition - Arousal Classification, 9-class task   |
| **Model Used**      | MLSTM-FCN                                                    |
| **Reference Paper** | [Multivariate LSTM-FCNs for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200) |
| **Original Code**   | https://github.com/titu1994/MLSTM-FCN (Original Model Repository)<br />https://github.com/athar70/MLSTM (DEAP Experiment) |

This experiment demonstrates how to use Tyee to perform a 9-class arousal classification task based on multi-modal peripheral physiological signals on the **DEAP** dataset using the **MLSTM-FCN** model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for multi-modal time series classification in Tyee.

------

## 2. Prerequisites

- **Download Location**: [DEAP](http://eecs.qmul.ac.uk/mmv/datasets/deap/)

- Directory Structure: Please download and decompress the dataset (the 

  `data_preprocessed_python`folder) and arrange it according to the following structure:

  ```
  /path/to/data/DEAP/
  └── data_preprocessed_python/
      ├── s01.dat
      ├── s02.dat
      └── ...
  ```

------

## 3. Model Configuration

This experiment uses the **MLSTM-FCN** model for arousal classification. Its key parameters are set in the configuration file as follows:

- **`max_nb_variables`**: 4 (corresponding to the 4 physiological signals: GSR, Resp, PPG, Temp)
- **`max_timesteps`**: 640 (the number of time points per sample)
- **`nb_class`**: 9 (corresponding to the 9 arousal levels)

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`HoldOut`)**: All trial data from all 32 subjects are pooled and then randomly split.
- **Split Ratio (`val_size`)**: **30%** of the data is allocated to the validation set, with the remaining **70%** serving as the training set.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is fully defined in the `offline_signal_transform` and `offline_label_transform` sections of the configuration file. The core steps include:

1. Signal Processing:
   - **Channel Concatenation (`Concat`)**: Merges the GSR, Resp, PPG, and Temp signals along the channel dimension.
   - **Normalization (`MinMaxNormalize`)**: Applies Min-Max normalization to the combined multi-modal signal.
   - **Sliding Window (`SlideWindow`)**: Segments the signal using a window size of 640 and a stride of 384 to generate samples.
2. Label Processing:
   - **Numerical Processing (`Round`, `ToNumpyInt32`)**: Rounds the original floating-point labels and converts them to integers.
   - **Label Mapping (`Mapping`)**: Maps the original label values of 1-9 to class indices of 0-8.

### 4.3 Task Definition

- **Task Type**: `deap_task.DEAPTask`
- **Core Logic**: This task is responsible for receiving the combined multi-modal signal `mulit4` and the processed arousal label `arousal`, performing a forward pass through the model, and computing the loss using **CrossEntropyLoss** to drive model training.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **1e-3**.
- **Learning Rate Scheduler (`ReduceLROnPlateauScheduler`)**: Monitors the training set loss; if the loss does not decrease for 100 consecutive epochs, the learning rate is multiplied by approximately 0.7.
- **Training Period**: Trains for a total of **2000 epochs**.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: deap_dataset.DEAPDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/DEAP/data_preprocessed_python'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_deap/train"
  io_mode: hdf5
  io_chunks: 640
  split: 
    select: HoldOut
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_deap/split
      val_size: 0.3
      random_state: 42
      shuffle: true
      
  offline_signal_transform:
    - select: Concat
      axis: 0
      source: ['gsr', 'resp', 'ppg', 'temp']
      target: mulit4
    - select: Compose
      transforms:
        - select: MinMaxNormalize
          axis: -1
        - select: SlideWindow
          window_size: 640
          stride: 384
      source: mulit4
      target: mulit4
    - select: Select
      key: ['mulit4']

  offline_label_transform:
    - select: Compose
      transforms:
        - select: Round
        - select: ToNumpyInt32
        - select: Mapping
          mapping:
            1: 0
            2: 1
            3: 2
            4: 3
            5: 4
            6: 5
            7: 6
            8: 7
            9: 8
      source: arousal
      target: arousal
    - select: Select
      key: ['arousal']

lr_scheduler:
  select: ReduceLROnPlateauScheduler
  patience_epochs: 100
  factor: 0.7071
  min_lr: 1e-4
  metric_source: train
  metric: loss

model:
  select: mlstm_fcn.MLSTM_FCN
  max_nb_variables: 4
  max_timesteps: 640
  nb_class: 9

optimizer:
  lr:  1e-3
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: deap_task.DEAPTask

trainer:
  fp16: false
  total_epochs: 2000
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, precision_macro, f1_macro, recall_macro]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/deap.yaml
   ```

   *(Please replace `config/deap.yaml` with your actual configuration file path.)*

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain validation performance similar to the table below. The evaluation methodology is as follows: the best model is trained on the training set and selected based on the **best accuracy** achieved on the validation set.

|          | Accuracy |
| -------- | -------- |
| Tyee     | 59.82    |
| Official | 47.3     |