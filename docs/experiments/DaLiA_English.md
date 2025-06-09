# PPG-DaLiA - Heart Rate Estimation

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | PPG-DaLiA                                                    |
| **Signal Type**     | PPG (Photoplethysmography), ACC (Tri-axial Accelerometer)    |
| **Analysis Task**   | Heart Rate Estimation, a regression task to predict continuous heart rate values from PPG and ACC signals. |
| **Model Used**      | BeliefPPG                                                    |
| **Reference Paper** | [BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation](https://proceedings.mlr.press/v216/bieri23a.html) |
| **Original Code**   | https://github.com/eth-siplab/BeliefPPG                      |

This experiment demonstrates how to use the Tyee framework to perform a heart rate estimation task on the **PPG-DaLiA** dataset using the **BeliefPPG** model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for multi-modal regression and handling complex task logic in Tyee.

------

## 2. Prerequisites

- **Download Location**: [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia)

- Directory Structure: Please download and decompress the dataset (the 

  `PPG_FieldStudy`folder) and arrange it according to the following structure:

  ```
  /path/to/data/ppg_dalia/
  └── PPG_FieldStudy/
      ├── S1/
      ├── S2/
      └── ...
  ```

------

## 3. Model Configuration

This experiment uses the **BeliefPPG** model, which will be instantiated with its default architecture settings.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment, including the complex data processing pipeline, model parameters, and training strategy, are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`LosoRotatingCrossSplit`)**: This experiment uses a rotating form of Leave-One-Subject-Out cross-validation. In each "fold" of the experiment, **one subject is held out as the test set**, **two subjects are held out as the validation set**, and all remaining subjects are used for training. By rotating through different combinations of subjects, the model is eventually trained and evaluated on all data.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is highly complex and is fully defined in the `offline_signal_transform` and `offline_label_transform` sections of the configuration file. The core steps include:

1. **Initial Windowing**: A first pass of `SlideWindow` is applied to the raw PPG and ACC signals to extract overlapping segments.
2. **Feature Extraction**: Within each signal segment, a series of transformations such as detrending, filtering, and normalization are executed, culminating in the generation of spectrogram features via `FFTSpectrum`.
3. **Feature Stacking & Secondary Windowing**: The spectrogram features from the PPG and ACC signals are stacked (`Stack`), and a second sliding window is applied to this new feature sequence to capture temporal-spectral dynamics.
4. **Time-Domain Feature Extraction**: Concurrently, a separate pipeline extracts time-domain features from the raw PPG signal.
5. Ultimately, the model receives both the processed **spectrogram sequence (`ppg_acc`)** and the **time-domain sequence (`ppg_time`)** as dual inputs.

### 4.3 Task Definition

- **Task Type**: `dalia_hr_task.DaLiaHREstimationTask`
- Core Logic:
  - **Fitting the Prior Layer (`on_train_start`)**: Before training officially begins, a hook is used to iterate through the entire training dataset, allowing the model's `PriorLayer` to learn and fit the prior distribution of all heart rate labels in the training set.
  - **Loss Function (`BinnedRegressionLoss`)**: A special "binned regression loss" is used, which transforms the continuous value prediction problem into a "classification-like" problem of predicting a probability distribution.
  - **Validation Step (`valid_step`)**: During validation, the model's `PriorLayer` is activated to refine the model's output by incorporating this prior knowledge.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **2.5e-4**.
- **Learning Rate Scheduler (`ReduceLROnPlateauScheduler`)**: Monitors the training set loss; if the loss does not decrease for 3 consecutive epochs, the learning rate is halved.
- **Evaluation Metric**: Uses **Mean Absolute Error (`mae`)** as the core evaluation metric, selecting the model with the lowest `mae` as the best model.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 2025
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: dalia_dataset.DaLiADataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/ppg_dalia/PPG_FieldStudy'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_ppgdalia/train"
  io_mode: hdf5
  io_chunks: 320
  split: 
    select: LosoRotatingCrossSplit
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_ppgdalia/split_official
      n_splits: 4
      group_by: subject_id
      shuffle: true
      random_state: 7

  offline_signal_transform:
    - select: Compose
      transforms:
        - select: SlideWindow
          window_size: 512
          stride: 128
        - select: WindowExtract
        - select: ForEach
          transforms:
            - select: Detrend
            - select: Filter
              l_freq: 0.4
              h_freq: 4
              method: iir
              phase: forward
              iir_params:
                order: 4
                ftype: butter
            - select: ZScoreNormalize
              axis: -1
              epsilon: 1e-10
            - select: Mean
              axis: 0
            - select: Resample
              desired_freq: 25
              window: boxcar
              pad: constant
              npad: 0
            - select: FFTSpectrum
              resolution: 535
              min_hz: 0.5
              max_hz: 3.5
      source: ppg
      target: ppg_spec
    - select: Compose
      transforms:
        - select: SlideWindow
          window_size: 256
          stride: 64
        - select: WindowExtract
        - select: ForEach
          transforms:
            - select: Detrend
            - select: Filter
              l_freq: 0.4
              h_freq: 4
              method: iir
              phase: forward
              iir_params:
                order: 4
                ftype: butter
            - select: ZScoreNormalize
              axis: -1
              epsilon: 1e-10
            - select: Resample
              desired_freq: 25
              window: boxcar
              pad: constant
              npad: 0
            - select: FFTSpectrum
              resolution: 535
              min_hz: 0.5
              max_hz: 3.5
              axis: -1
            - select: Mean
              axis: 0
      source: acc
      target: acc_spec
    - select: Stack
      axis: -1
      source: [ppg_spec, acc_spec]
      target: ppg_acc
    - select: Compose
      transforms:
        - select: ZScoreNormalize
          epsilon: 1e-10
        - select: SlideWindow
          window_size: 7
          stride: 1
          axis: 0
      source: ppg_acc
      target: ppg_acc
    - select: Compose
      transforms:
        - select: Detrend
        - select: Filter
          l_freq: 0.1
          h_freq: 18
          method: iir
          phase: forward
          iir_params:
            order: 4
            ftype: butter
        - select: Mean
          axis: 0
        - select: ExpandDims
          axis: -1
        - select: ZScoreNormalize
          epsilon: 1e-10
        - select: SlideWindow
          window_size: 1280
          stride: 128
          axis: 0
      source: ppg
      target: ppg_time
    - select: Select
      key: [ppg_acc, ppg_time]

  offline_label_transform:
    - select: Compose
      transforms:
        - select: Crop
          crop_left: 6
        - select: SlideWindow
          window_size: 1
          stride: 1
          axis: 0
      source: hr
      target: hr

lr_scheduler:
  select: ReduceLROnPlateauScheduler
  patience_epochs: 3
  factor: 0.5
  min_lr: 1e-6
  metric_source: train
  metric: loss

model:
  select: beliefppg.beliefppg.BeliefPPG

optimizer:
  lr: 2.5e-4
  select: Adam

task:
  loss:
    select: BinnedRegressionLoss
    dim: 64
    min_hz: 0.5
    max_hz: 3.5
    sigma_y: 1.5
  select: dalia_hr_task.DaLiaHREstimationTask

trainer:
  fp16: true
  total_epochs: 50
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: mae
    mode: min
  metrics: [mae, r2]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. Run the Experiment: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/dalia.yaml
   ```

   (Please replace `config/dalia.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. The evaluation methodology is as follows: in **each fold** of the cross-validation, the best model is selected based on the **lowest Mean Absolute Error (best MAE)** on the validation set, and then evaluated on the corresponding test set for that fold. The table shows the **average performance of the test results across all folds**.

|          | Mean Absolute Error (MAE) |
| -------- | ------------------------- |
| Tyee     | 4.22                      |
| Official | 4.02                      |

**Note**: Data augmentation was used in the original BeliefPPG paper's experiment, but it was not used in this replication.