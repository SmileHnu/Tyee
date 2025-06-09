# TUAB - Abnormal EEG Detection

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | TUH Abnormal EEG Corpus (TUAB) v3.0.1                        |
| **Signal Type**     | EEG                                                          |
| **Analysis Task**   | Abnormal EEG Detection, a binary classification task to distinguish between normal and abnormal EEG recordings. |
| **Model Used**      | LaBraM (Large Brain Model)                                   |
| **Reference Paper** | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) |
| **Original Code**   | https://github.com/935963004/LaBraM                          |

This experiment aims to use Tyee to perform a **binary classification of abnormal EEG signals** on the **TUAB** dataset by fine-tuning the **LaBraM** pre-trained model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for applying large pre-trained models for classification in Tyee.

------

## 2. Prerequisites

- **Download Location**: [TUAB](https://www.google.com/search?q=https://isip.piconepress.com/projects/nedc/html/tuh_eeg/%23c_tuab) (Requires application and authorization to access TUAB)

- **Directory Structure**: Please download and decompress the dataset, keeping the original structure as follows:

  ```
  /path/to/data/tuh_eeg_abnormal/v3.0.1/edf/
  ├── train/
  │   ├── abnormal/
  │   └── normal/
  └── eval/
      ├── abnormal/
      └── normal/
  ```

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the **LaBraM** (`labram_base_patch200_200`) model for abnormal EEG detection.

### 3.2 Model Parameters

The key parameters for this model in this experiment are set in the configuration file as follows:

- **`nb_classes`**: 2 (Corresponding to normal/abnormal binary classification)
- **`drop`**: 0.0 (Dropout rate in the fully connected layers)
- **`drop_path`**: 0.1 (Drop Path rate for Stochastic Depth)
- **`attn_drop_rate`**: 0.0 (Dropout rate in the attention mechanism)
- **`use_mean_pooling`**: true (Uses average pooling to aggregate features)
- **`layer_scale_init_value`**: 0.1 (Initial value for Layer Scale)

### 3.3 Pre-trained Weights Information

In this experiment, we load the **official upstream pre-trained weights** for **LaBraM** as the starting point for model initialization and then fine-tune it on the TUAB dataset.

- **Weight Source**: The pre-trained weights are provided by the original model authors.
- **Download Link**: https://github.com/935963004/LaBraM/tree/main/checkpoints
- **Usage**: Please place the downloaded weight file at the path specified by `model.finetune` in the configuration file. Tyee will automatically load these weights at the start of the experiment.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`HoldOutCross`)**: This experiment follows the official training and test set (used here as the `eval` set) split of the TUAB dataset. Building on this, we further divide the official **training set** into a new training set and a validation set.
- **Split Ratio (`val_size`)**: From the original training set, **20%** of the subjects are allocated to the validation set, grouped by subject ID (`group_by: subject_id`), with the remaining **80%** serving as the new training set.

### 4.2 Data Processing Pipeline

1. Before-Segment Transform (`before_segment_transform`):
   - **Channel Selection (`PickChannels`)**: Selects 23 standard EEG channels.
   - **Filtering (`Filter` & `NotchFilter`)**: Applies a 0.1-75 Hz band-pass filter and a 50 Hz notch filter to remove power-line interference.
   - **Resampling (`Resample`)**: Resamples the signal to a uniform sampling rate of 200 Hz.
2. Offline Transform (`offline_signal_transform`):
   - **Sliding Window (`SlideWindow`)**: Segments the continuous signal into non-overlapping windows of 2000 time points (10 seconds) with a stride of 2000.
3. Label Transform (`offline_label_transform`):
   - **Label Mapping (`Mapping`)**: Maps the 'abnormal' and 'normal' labels to 1 and 0, respectively.

### 4.3 Task Definition

- **Task Type**: `tuab_task.TUABTask`
- Core Logic:
  - **Model Building (`build_model`)**: Responsible for loading the LaBraM model architecture and its official pre-trained weights from the specified path.
  - **Optimizer Parameter Setup (`set_optimizer_params`)**: Implements a **Layer-wise Learning Rate Decay** strategy, applying different learning rates to layers at different depths of the model.
  - **Train/Validation Step**: Before feeding data to the model, it performs a specific `rearrange` operation to match LaBraM's input format and scales the input signal by `/100`.

### 4.4 Training Strategy

- **Optimizer**: Uses **AdamW** with a base learning rate of **5e-4**, a weight decay of **0.05**, and a layer-wise learning rate decay of **0.65**.
- **Learning Rate Scheduler (`CosineLRScheduler`)**: Employs a cosine annealing learning rate over a total of 50 epochs, which includes a 5-epoch linear warm-up phase.
- **Loss Function**: Uses **LabelSmoothingCrossEntropy** with a smoothing factor of 0.1.
- **Gradient Accumulation**: The `update_interval` is set to 8, meaning gradients are accumulated over 8 batches before a model parameter update is performed.

### 4.5 Full Configuration File

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: tuab_dataset.TUABDataset
  num_workers: 18
  root_path:
    train: '/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/train'
    test: '/mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/eval'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_tuab/processed_train"
    test: "/mnt/ssd/lingyus/tyee_tuab/processed_eval"
  io_mode: hdf5
  split: 
    select: HoldOutCross
    init_params:
      split_path: /mnt/ssd/lingyus/tuh_eeg_abnormal/v3.0.1/edf/split
      group_by: subject_id
      val_size: 0.2
      random_state: 12345
      shuffle: true

  before_segment_transform:
    - select: Compose
      transforms:
        - select: PickChannels
          channels: ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        - select: Filter
          l_freq: 0.1
          h_freq: 75.0
        - select: NotchFilter
          freqs: [50.0]
        - select: Resample
          desired_freq: 200
      source: eeg
      target: eeg
  
  offline_signal_transform:
    - select: SlideWindow
      window_size: 2000
      stride: 2000
      source: eeg
      target: eeg

  offline_label_transform:
    - select: Mapping
      mapping:
        abnormal: 1
        normal: 0
      source: label
      target: label

lr_scheduler:
  select: CosineLRScheduler
  period_epochs: 50
  min_lr: 1e-6
  warmup_start_lr: 0
  warmup_epochs: 5

model:
  select: labram.labram_base_patch200_200
  finetune: /home/lingyus/code/PRL/models/labram/checkpoints/labram-base.pth
  nb_classes: 2
  use_mean_pooling: true

optimizer:
  lr: 5e-4
  select: AdamW
  weight_decay: 0.05
  layer_decay: 0.65

task:
  loss:
    select: LabelSmoothingCrossEntropy
    smoothing: 0.1
  select: tuab_task.TUABTask

trainer:
  fp16: true
  total_epochs: 50
  update_interval: 8
  log_interval: 20
  eval_metric:
    select: balanced_accuracy
    mode: max
  metrics: [balanced_accuracy, accuracy, pr_auc, roc_auc]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) and the pre-trained model path (`model.finetune`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/tuab.yaml
   ```

   *(Please replace `config/tuab.yaml` with your actual configuration file path.)*

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. The evaluation methodology is as follows: the best model is trained on the training set, selected based on the **best balanced accuracy** achieved on the validation set, and finally evaluated on the test set.

|          | Balanced Accuracy | AUC-PR | AUROC |
| -------- | ----------------- | ------ | ----- |
| Tyee     | 82.37             | 90.96  | 90.65 |
| Official | 81.40             | 89.65  | 90.22 |