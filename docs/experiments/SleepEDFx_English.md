# Sleep-EDF - Sleep Staging

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | Sleep-EDF Expanded (Sleep-EDF-39), SC-subjects subset        |
| **Signal Type**     | EEG (Fpz-Cz), EOG (horizontal)                               |
| **Analysis Task**   | Sleep Staging, 5-class task (W, N1, N2, N3, R)               |
| **Model Used**      | SalientSleepNet                                              |
| **Reference Paper** | [SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://arxiv.org/abs/2105.13864) |
| **Original Code**   | https://github.com/ziyujia/SalientSleepNet                   |

This experiment aims to use Tyee to perform a **sleep staging** task on the **Sleep-EDF-39** dataset using the **SalientSleepNet** model, based on EEG and EOG signals. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for multi-modal time-series data analysis in Tyee.

------

## 2. Prerequisites

- **Download Location**: [Sleep-EDF Database on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)

- **Dataset Subset Description**: This experiment uses the first 20 subjects from the Sleep Cassette (SC) recordings of the Sleep-EDF Expanded dataset. This subset is also known as Sleep-EDF-20 (for its 20 subjects) or Sleep-EDF-39 (for its 39 

  `.edf`files). You will need to extract the `.edf`files (both PSG and Hypnogram) for the following 20 subjects from the downloaded 

  `sleep-cassette`folder:

  - `SC4001`, `SC4011`, `SC4021`, `SC4031`, `SC4041`, `SC4051`, `SC4061`, `SC4071`, `SC4081`, `SC4091`, `SC4101`, `SC4111`, `SC4121`, `SC4131`, `SC4141`, `SC4151`, `SC4161`, `SC4171`, `SC4181`, `SC4191`

- **Directory Structure**: Please arrange the extracted files according to the following structure:

  ```
  /path/to/data/sleep-edf-39/
  ├── SC4001E0-PSG.edf
  ├── SC4001EC-Hypnogram.edf
  ├── SC4011E0-PSG.edf
  └── ...
  ```

------

## 3. Model Configuration

This experiment uses the **SalientSleepNet** (`TwoStreamSalientModel`) model. Its key architectural parameters are set in the `model.config` section of the configuration file as follows:

- `sleep_epoch_len`: 3000 (Length of each sleep epoch, 30s * 100Hz)
- `sequence_epochs`: 20 (Number of consecutive sleep epochs included in each input sample)
- `filters`: [16, 32, 64, 128, 256] (Number of filters at each stage of the U-Net encoder)
- `kernel_size`: 5 (Size of the convolutional kernels)
- `pooling_sizes`: [10, 8, 6, 4] (Pooling sizes at each stage of the U-Net encoder)
- `u_depths`: [4, 4, 4, 4] (Depth of the U-Net units)

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`KFoldCross`)**: A 20-fold cross-validation is applied to the 20 subjects in the dataset, which is equivalent to Leave-One-Subject-Out cross-validation. In each fold, one subject's data is used as the validation set, while the data from the remaining 19 subjects is used for training.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is defined in the configuration file and is based on the implementation from the public repository **SleepDG**.

- **Preprocessing Reference**: [SleepDG (GitHub)](https://github.com/wjq-learning/SleepDG)
- **Core Steps**:
  1. **Channel Selection (`PickChannels`)**: Selects only the 'Fpz-Cz' lead for the EEG signal.
  2. **Label Mapping (`Mapping`)**: Maps the original sleep stages (including N4) to the standard 5 classes (W, N1, N2, N3, R).
  3. **Contextual Windowing (`SlideWindow`)**: Combines each 30-second sleep epoch with its 9 preceding and 10 succeeding epochs to form a long sequence of 20 epochs as a single input sample for the model.
- **A Note on Sample Count**: Due to potential minor differences in the preprocessing pipeline compared to the original paper, the resulting number of labels in this experiment differs slightly from what was reported in the paper.
  - **Original Paper Label Count**: {W: 8285, N1: 2804, N2: 17799, N3: 5703, REM: 7717}
  - **This Experiment's Label Count**: {W: 7619, N1: 2804, N2: 17799, N3: 5703, REM: 7715}

### 4.3 Task Definition

- **Task Type**: `sleepedfx_task.SleepEDFxTask`
- **Core Logic**:
  - **Loss Function**: Uses a weighted **CrossEntropyLoss**, with weights `[1.0, 1.80, 1.0, 1.20, 1.25]`, to mitigate the common class imbalance problem in sleep staging tasks.
  - **Train/Validation Step**: Receives long sequences of EEG and EOG as dual inputs and gets predictions from the model. Since the model predicts 20 epochs at a time, the labels and predictions are flattened (`reshape`) before the loss is calculated.

### 4.4 Training Strategy

- **Optimizer**: Uses Adam with a learning rate of 0.001.
- **Training Period**: Trains for a total of 60 epochs.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model.

### 4.5 Full Configuration File

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 12
  dataset: sleepedfx_dataset.SleepEDFCassetteDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/sleep-edf-20'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_sleepedfx_20/train"
  io_mode: hdf5
  io_chunks: 20
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_sleepedfx_20/split_17_20
      group_by: subject_id
      n_splits: 20
      shuffle: false

  before_segment_transform:
    - select: PickChannels
      channels: ['Fpz-Cz']
      source: eeg
      target: eeg
  
  offline_signal_transform:
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: eeg
      target: eeg
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: eog
      target: eog
    - select: Select
      key: ['eeg', 'eog']
  
  offline_label_transform:
    - select: Mapping
      mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4} # N4 -> N3
      source: stage
      target: stage
    - select: SlideWindow
      window_size: 20
      stride: 20
      axis: 0
      source: stage
      target: stage
  
  online_signal_transform:
    - select: Compose
      transforms:
        - select: Transpose
          axes: [1, 0, 2]
        - select: Reshape
          shape: [1, -1]
        - select: ExpandDims
          axis: -1
      source: eeg
      target: eeg
    - select: Compose
      transforms:
        - select: Transpose
          axes: [1, 0, 2]
        - select: Reshape
          shape: [1, -1]
        - select: ExpandDims
          axis: -1
      source: eog
      target: eog

model:
  select: salient_sleep_net.TwoStreamSalientModel
  config:
    sleep_epoch_len: 3000
    preprocess:
      sequence_epochs: 20
    train:
      filters: [16, 32, 64, 128, 256]
      kernel_size: 5
      pooling_sizes: [10, 8, 6, 4]
      dilation_sizes: [1, 2, 3, 4]
      activation: 'relu'
      u_depths: [4, 4, 4, 4]
      u_inner_filter: 16
      mse_filters: [8, 16, 32, 64, 128]
      padding: 'same'

optimizer:
  lr:  0.001
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
    weight: [1.0, 1.80, 1.0, 1.20, 1.25]
  select: sleepedfx_task.SleepEDFxTask

trainer:
  fp16: false
  total_epochs: 60
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
   python main.py --config config/sleep_edf.yaml
   ```

   (Please replace `config/sleep_edf.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain validation performance similar to the table below. As this experiment uses 20-fold cross-validation, the table shows the **average performance on the validation set across all folds**.

|          | Accuracy | F1 (Macro) |
| -------- | -------- | ---------- |
| Tyee     | 88.25    | 82.77      |
| Official | 87.5     | 83         |