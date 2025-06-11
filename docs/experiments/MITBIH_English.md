# MIT-BIH - Arrhythmia Classification

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | MIT-BIH Arrhythmia Database                                  |
| **Signal Type**     | ECG (Electrocardiogram)                                      |
| **Analysis Task**   | Arrhythmia Classification, a multi-class classification task to distinguish between different heartbeat types. |
| **Model Used**      | EcgResNet34                                                  |
| **Reference Paper** | [Diagnosis of Diseases by ECG Using Convolutional Neural Networks](https://www.hse.ru/en/edu/vkr/368722189) |
| **Original Code**   | https://github.com/lxdv/ecg-classification                   |

This experiment demonstrates how to use Tyee to perform an arrhythmia classification task on the **MIT-BIH** dataset using a custom **EcgResNet34** model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for ECG signal classification in Tyee.

------

## 2. Prerequisites

- **Download Location**: [MIT-BIH Arrhythmia Database on PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

- **Directory Structure**: Please download and decompress the dataset and arrange it according to the following structure:

  ```
  /path/to/data/mit-bih-arrhythmia-database-1.0.0/
  ├── 100.atr
  ├── 100.dat
  ├── 100.hea
  └── ...
  ```

------

## 3. Model Configuration

This experiment uses the **EcgResNet34** model, which will be instantiated with its default architecture settings.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`HoldOut`)**: All heartbeat samples from all records are pooled and then split using **stratified random sampling** (`stratify: symbol`). This strategy ensures that the proportion of different arrhythmia types is approximately the same in both the training and validation sets as it is in the original dataset.
- **Split Ratio (`val_size`)**: **10%** of the data is allocated to the validation set, with the remaining **90%** serving as the training set.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is fully defined in the `before_segment_transform` and `offline_label_transform` sections of the configuration file. The core steps include:

1. Signal (ECG) Processing:
   - **Channel Selection (`PickChannels`)**: Only the 'MLII' lead signal is selected for analysis.
   - **Normalization (`ZScoreNormalize`)**: The signal is normalized using Z-score normalization.
2. Label (Symbol) Processing:
   - **Label Selection and Mapping (`Mapping`)**: From all arrhythmia types in the MIT-BIH dataset, 8 major types are selected for the classification task. The following mapping relationship is used to convert character labels to numerical class indices:
     ```
     'N': 0  # Normal beat
     'V': 1  # Premature ventricular contraction
     '/': 2  # Paced beat
     'R': 3  # Right bundle branch block beat
     'L': 4  # Left bundle branch block beat
     'A': 5  # Atrial premature beat
     '!': 6  # Ventricular flutter wave
     'E': 7  # Ventricular escape beat
     ```
   - **Label Filtering**: Through the Mapping operation, arrhythmia type labels other than the above 8 types are automatically filtered out, ensuring the classification task focuses on these 8 major categories.

### 4.3 Task Definition

- **Task Type**: `mit_bih_task.MITBIHTask`
- **Core Logic**: This task is responsible for receiving the ECG signal `x` and the arrhythmia type label `symbol`, performing a forward pass through the EcgResNet34 model, and computing the loss using **CrossEntropyLoss** to drive model training.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **0.001**.
- **Training Period**: Trains for a total of **650 epochs**.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: mit_bih_dataset.MITBIHDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/mit-bih-arrhythmia-database-1.0.0'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_mit_bih/train"
  io_mode: hdf5
  io_chunks: 128
  split: 
    select: HoldOut
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_mit_bih/split
      val_size: 0.1
      random_state: 7
      shuffle: true
      stratify: symbol

  before_segment_transform:
    - select: PickChannels
      channels: ['MLII']
      source: ecg
      target: ecg
    - select: ZScoreNormalize
      axis: 1
      source: ecg
      target: ecg
      
  offline_label_transform:
    - select: Mapping
      mapping:
        'N': 0
        'V': 1
        '/': 2
        'R': 3
        'L': 4
        'A': 5
        '!': 6
        'E': 7
      source: symbol
      target: symbol

model:
  select: ecgresnet34.EcgResNet34

optimizer:
  lr:  0.001
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: mit_bih_task.MITBIHTask

trainer:
  fp16: false
  total_epochs: 650
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

   ```yaml
   python main.py --config config/mit_bih.yaml
   ```

   (Please replace `config/mit_bih.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain validation performance similar to the table below. The evaluation methodology is as follows: the best model is trained on the training set and selected based on the **best accuracy** achieved on the validation set.

|          | Accuracy |
| -------- | -------- |
| Tyee     | 99.51    |
| Official | 99.38    |