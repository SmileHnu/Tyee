# BCI Competition IV Dataset 4 - Finger Movement Decoding

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | BCI Competition IV Dataset 4                                 |
| **Signal Type**     | ECoG (Electrocorticography)                                  |
| **Analysis Task**   | Finger Movement Decoding, a regression task to predict the flexion of five fingers. |
| **Model Used**      | FingerFlex (AutoEncoder1D)                                   |
| **Reference Paper** | [FingerFlex: Inferring Finger Trajectories from ECoG signals](https://arxiv.org/abs/2211.01960) |
| **Original Code**   | https://github.com/Irautak/FingerFlex                        |
| **Special Note**    | This experiment is conducted on a single-subject basis.      |

This experiment demonstrates how to use Tyee to decode finger movement trajectories from ECoG signals on the **BCICIV 4** dataset using the **FingerFlex** model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for regression tasks in brain-computer interfaces with Tyee.

------

## 2. Prerequisites

- **Download Location**: [BCICIV-4](https://www.bbci.de/competition/iv/#datasets)

- Directory Structure: Please download and decompress the dataset and arrange it according to the following structure:

  ```
  /path/to/data/BCICIV4/
  └── sub1/
      ├── sub1_comp_ecog.mat
      ├── sub1_comp_dg.mat
      └── ... (other subjects)
  ```

------

## 3. Model Configuration

This experiment uses the **FingerFlex (`AutoEncoder1D`)** model for finger movement decoding. The model is **trained from scratch** in this experiment. Its key architectural parameters are set in the configuration file as follows:

- **`channels`**: `[32, 32, 64, 64, 128, 128]` (The number of output channels for each convolutional layer in the encoder)
- **`kernel_sizes`**: `[7, 7, 5, 5, 5]` (The kernel size for each convolutional layer in the encoder)
- **`strides`**: `[2, 2, 2, 2, 2]` (The stride for each convolutional layer in the encoder)
- **`n_electrodes`**: 62 (The number of input ECoG electrodes)
- **`n_freqs`**: 40 (The number of spectral features per electrode)
- **`n_channels_out`**: 5 (The number of final output channels, corresponding to the five fingers)

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy**: This experiment follows the official dataset split. Since the official training and test sets (used here as a validation set) are contained within a single file, we treat them as two separate trials. In the configuration file, we use the `HoldOutCross` strategy and set `group_by: trial_id` and `val_size: 0.5` to precisely use one trial for training and the other for validation.

### 4.2 Data Processing Pipeline

**Important Note**: Due to the complexity of the data preprocessing pipeline for this experiment, it is implemented directly in the `dataset` class (`bciciv4_dataset.py`) and is **not defined in the `config.yaml` file**. The detailed transformation code can be found in **Appendix A.1**.

The main steps are summarized as follows:

1. **Signal (ECoG) Processing**: Z-score normalization, Common Average Referencing (CAR), band-pass filtering, notch filtering, continuous wavelet transform (CWT) for spectral feature extraction, downsampling, robust normalization, and sliding window.
2. **Label (Fingers) Processing**: Downsampling, interpolation, Min-Max normalization, and sliding window.

### 4.3 Task Definition

- **Task Type**: `bciciv4_task.BCICIV4Task`
- Core Logic:
  - **Loss Function**: A composite loss function is used, defined as **0.5 \* MSE_Loss + 0.5 \* (1 - Correlation)**. This loss function simultaneously optimizes for the mean squared error and the correlation between the predicted and true trajectories.
  - **Train/Validation Step**: Receives the ECoG signal `x` and finger data `label`, gets the prediction `pred` from the model, and then computes the loss using the composite loss function described above.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **8.42e-5** and a weight decay of `1e-6`.
- **Training Period**: Trains for a total of **20 epochs**.
- **Evaluation Metric**: Uses the **mean correlation coefficient (`mean_cc`)** as the core evaluation metric and selects the model with the highest `mean_cc` during training as the best model.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: bciciv4_dataset.BCICIV4Dataset
  num_workers: 8
  root_path:
    train: '/home/lingyus/data/BCICIV4/sub1'
  io_path:
    train: "/home/lingyus/data/BCICIV4/sub1/processed_test"
  io_mode: hdf5
  split:
    select: HoldOutCross
    init_params:
      split_path: /home/lingyus/data/BCICIV4/sub1/split
      group_by: trial_id
      val_size: 0.5
      random_state: 0
      shuffle: true

model:
  select: fingerflex.AutoEncoder1D
  channels: [32, 32, 64, 64, 128, 128]
  kernel_sizes: [7, 7, 5, 5, 5]
  strides: [2, 2, 2, 2, 2]
  dilation: [1, 1, 1, 1, 1]
  n_electrodes: 62
  n_freqs: 40
  n_channels_out: 5

optimizer:
  lr:  8.42e-5
  select: Adam
  weight_decay: 1e-6

task:
  loss:
    select: MSELoss
  select: bciciv4_task.BCICIV4Task

trainer:
  fp16: false
  total_epochs: 20
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: mean_cc
    mode: max
  metrics: [mean_cc]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/bciciv4.yaml
   ```

   (Please replace `config/bciciv4.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs, including **training logs**, **model weights (.pt files)**, and **TensorBoard results**, will be saved in the directory specified by `common.exp_dir` in the configuration file, within a timestamped subfolder (e.g., `./experiments/2025-06-07/12-00-00/`).

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain final validation performance similar to the table below. The table shows the **best performance** achieved by the model on the validation set for **subject S1** during the entire training process.

|          | Mean Correlation Coefficient |
| -------- | ---------------------------- |
| Tyee     | 0.6925                       |
| Official | 0.66                         |

------

## Appendix

### A.1 Data Preprocessing Code & Normalization Statistics

The data preprocessing pipeline for this experiment is defined directly in `bciciv4_dataset.py`.

**A Note on Normalization Statistics (`robust_stats` and `minmax_stats`):** To prevent information leakage from the validation set, we follow a strict normalization procedure:

1. First, run all preprocessing steps on the dataset **except for** `RobustNormalize` and `MinMaxNormalize`.
2. Then, split the processed data into training and validation sets.
3. Calculate the `center_` (median) and `scale_` (interquartile range) for robust normalization, and the `data_min_` and `data_max_` for Min-Max normalization, **using only the training set**. These statistics are then saved to files (e.g., `.npz`).
4. Finally, in the complete processing pipeline, the `RobustNormalize` and `MinMaxNormalize` steps load these pre-calculated statistics from the files to transform both the training and validation sets, ensuring a consistent standard is used for both.

The core transformation logic is as follows:

```python
# Import required transform classes
from tyee.dataset.transform import (
    Compose, NotchFilter, Filter, ZScoreNormalize, RobustNormalize, Reshape,
    CWTSpectrum, Downsample, Crop, Interpolate, MinMaxNormalize, CommonAverageRef,
    Transpose, SlideWindow
)
import numpy as np

# Note: The following statistics are pre-calculated from the training set and loaded from files.
# robust_stats = np.load('/path/to/your/robust_scaler_stats0.npz')
# minmax_stats = np.load('/path/to/your/minmax_scaler_stats0.npz')

# --- Signal (ECoG) Preprocessing Transforms ---
offline_signal_transform = [
    ZScoreNormalize(epsilon=0, axis=1, source='ecog', target='ecog'),
    CommonAverageRef(axis=0, source='ecog', target='ecog'),
    Filter(l_freq=40, h_freq=300, source='ecog', target='ecog'),
    NotchFilter(freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450], source='ecog', target='ecog'),
    CWTSpectrum(freqs=np.logspace(np.log10(40), np.log10(300), 40), output_type='power', n_jobs=6, source='ecg', target='ecg'),
    Downsample(desired_freq=100, source='ecog', target='ecog'),
    Crop(crop_right=20, source='ecog', target='ecog'),
    Transpose(source='ecog', target='ecog'),
    Reshape(shape=(-1, 40*62), source='ecog', target='ecog'),
    RobustNormalize(
        median=robust_stats['center_'], iqr=robust_stats['scale_'], 
        unit_variance=False, quantile_range=(0.1, 0.9), epsilon=0, axis=0, source='ecog', target='ecog'
    ),
    Reshape(shape=(-1, 40, 62), source='ecog', target='ecog'),
    Transpose(source='ecog', target='ecog'),
    SlideWindow(window_size=256, stride=1, source='ecog', target='ecog'),
]

# --- Label (Fingers) Preprocessing Transforms ---
offline_label_transform = [
    Downsample(desired_freq=25, source='dg', target='dg'),
    Interpolate(desired_freq=100, kind='cubic', source='dg', target='dg'),
    Crop(crop_left=20, source='dg', target='dg'),
    Transpose(source='dg', target='dg'),
    MinMaxNormalize(
        min=minmax_stats['data_min_'], max=minmax_stats['data_max_'], 
        axis=0, source='dg', target='dg'
    ),
    Transpose(source='dg', target='dg'),
    SlideWindow(window_size=256, stride=1, source='dg', target='dg'),
]
```

### A.2 Special Note on Validation Set Windowing

The original FingerFlex code applies sliding window processing only to the training set, while the validation set is evaluated as a whole to get a continuous prediction that matches the length of the ground truth label.

To implement this asymmetric processing in the Tyee framework, we adopted the following strategy:

1. **Uniform Processing & Caching**: During the data loading and caching phase, both the training and validation sets are segmented into numerous overlapping windows by `SlideWindow` to ensure a consistent data processing pipeline.
2. **Sequence Restoration for Validation**: During **validation**, we use special evaluation logic that **ignores** the windowed structure of the validation set data. The framework is instructed to treat the complete, unsegmented validation `trial` as a single continuous long sequence for model evaluation.
3. **Aligned Evaluation**: This way, the model outputs a single, long continuous prediction for the entire validation `trial`, which can be directly compared with the full-length ground truth label to calculate the correlation.