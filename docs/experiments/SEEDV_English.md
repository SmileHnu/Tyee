# SEED-V - Emotion Recognition

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | SEED-V                                                       |
| **Signal Type**     | EEG-DE (Differential Entropy), EM-DE (Eye Movement Differential Entropy) |
| **Analysis Task**   | Emotion Recognition, 5-class task (happy, sad, neutral, fear, disgust) |
| **Model Used**      | G2G-ResNet18 (Graph-to-Grid with ResNet-18 backbone)         |
| **Reference Paper** | [Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://dl.acm.org/doi/abs/10.1145/3581783.3612074) |
| **Original Code**   | https://github.com/Jinminbox/G2G                             |

This experiment aims to use Tyee to perform a **5-class emotion recognition** task on the **DE features** of the **SEED-V** dataset, using the **Graph-to-Grid (G2G)** method. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for processing graph-structured EEG data and multi-modal feature fusion in Tyee.

------

## 2. Prerequisites

- **Download Location**: [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html)

- **Directory Structure**: Please download and decompress the dataset, keeping the original structure as follows:

  ```
  /path/to/data/SEED-V/
  ├── EEG_DE_features/
  │   ├── 1_123.npz
  │   ├── 2_123.npz
  │   └── ...
  └── Eye_movement_features/
      ├── 1_123.npz
      ├── 2_123.npz
      └── ...
  ```

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the **G2G-ResNet18** model. The core idea of this method is to use a `RelationAwareness` module to dynamically map graph-structured EEG features, based on the physical locations of electrodes, into a 2D grid (image) representation that preserves spatial topology. Subsequently, the generated image is fed into a classic **ResNet-18** backbone for feature extraction and classification.

### 3.2 Model Parameters

The key model parameters are set in the configuration file as follows:

- **`head_num`**: 6 (Number of attention heads in the relation-awareness module)
- **`rand_ali_num`**: 2 (Number of random permutation branches)
- **`backbone`**: "ResNet18" (The backbone for image processing)
- **`input_size`**: 5 (Feature dimension for each EEG electrode, i.e., DE values for 5 frequency bands)
- **`location_size`**: 3 (Dimension of the 3D spatial coordinates for electrodes)
- **`expand_size`**: 10 (The embedding dimension for features and location information)
- **`num_class`**: 5 (The number of final emotion classes for classification)

This model is **trained from scratch** in this experiment.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`KFoldPerSubjectCross`)**: This experiment uses **within-subject 3-fold cross-validation**. For each subject's data, the trials (`trial_id`) are split into 3 folds. In rotation, one fold is used as the validation set, while the other two are used for training.

### 4.2 Data Processing Pipeline

Since this experiment uses pre-extracted features, the data processing pipeline is relatively simple and is primarily defined in the `online_signal_transform` section:

1. **Normalization (`MinMaxNormalize`)**: Applies Min-Max normalization to the EEG and EOG features separately.
2. **Feature Concatenation (`Concat`)**: Merges the normalized EEG and EOG features.
3. **Feature Alignment (`Insert`)**: Inserts zero-values at specific locations to align the feature dimensions as required by the model.
4. **Squeeze**: Removes unnecessary dimensions to match the model's input shape.

### 4.3 Task Definition

- **Task Type**: `seedv_task.SEEDVFeatureTask`
- **Core Logic**:
  - **Optimizer Parameter Setup (`set_optimizer_params`)**: This method groups the model's parameters into different sets (G2G module, backbone, classification head), which facilitates differential training strategies in the future.
  - **Train/Validation Step**: Receives the processed multi-modal features `eeg_eog` and the emotion label `emotion`, performs a forward pass through the model, and computes the loss using **LabelSmoothingCrossEntropy**.

### 4.4 Training Strategy

- **Optimizer**: Uses **AdamW** with a learning rate of **0.008** and a weight decay of `5e-4`.
- **Learning Rate Scheduler (`StepLRScheduler`)**: Employs a step-wise learning rate decay schedule with a 2000-step linear warm-up phase at the beginning.
- **Training Period**: Trains for a total of **300 epochs**.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 222
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 32
  dataset: seedv_dataset.SEEDVFeatureDataset
  num_workers: 8
  root_path:
    train: '/mnt/ssd/lingyus/SEED-V'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_seedv_feature/train"
  io_mode: hdf5
  io_chunks: 1
  split: 
    select: KFoldPerSubjectCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_seedv_feature/split
      group_by: trial_id
      n_splits: 3
      shuffle: false

  offline_signal_transform:
    - select: Log
      epsilon: 1
      source: eog
      target: eog
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: eeg
      target: eeg
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: eog
      target: eog

  offline_label_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: emotion
      target: emotion

  online_signal_transform:
    - select: MinMaxNormalize
      source: eeg
      target: eeg
    - select: MinMaxNormalize
      source: eog
      target: eog
    - select: Concat
      axis: -1
      source: ["eeg", "eog"]
      target: eeg_eog
    - select: Insert
      indices: [316, 317, 318, 319, 326, 327, 328, 329, 334, 335, 336, 337, 338, 339, 344, 345, 346, 347, 348, 349, 354, 355, 356, 357, 358, 359, 369]
      value: 0
      axis: -1
      source: eeg_eog
      target: eeg_eog
    - select: Squeeze
      axis: 0
      source: eeg_eog
      target: eeg_eog
    - select: Select
      key: ["eeg_eog"]

  online_label_transform:
    - select: Squeeze
      axis: 0
      source: emotion
      target: emotion
    - select: ToNumpyInt64
      source: emotion
      target: emotion
  
lr_scheduler:
  select: StepLRScheduler
  gamma: 0.1
  epoch_size: 300
  warmup_steps: 2000 

model:
  select: g2g.EncoderNet
  head_num: 6
  rand_ali_num: 2
  backbone: "ResNet18"
  input_size: 5
  location_size: 3
  expand_size: 10
  eeg_node_num: 62
  num_class: 5
  sup_node_num: 6

optimizer:
  lr:  0.008
  select: AdamW
  weight_decay: 5e-4
  betas: (0.9, 0.999)

task:
  loss:
    select: LabelSmoothingCrossEntropy
    smoothing: 0.01
  select: seedv_task.SEEDVFeatureTask

trainer:
  fp16: false
  total_epochs: 300
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, balanced_accuracy,precision_macro, f1_macro, recall_macro]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/seedv.yaml
   ```

   (Please replace `config/seedv.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain validation performance similar to the table below. As this experiment uses within-subject 3-fold cross-validation, the table shows the **average performance on the validation set across all subjects and all folds**.

|          | Accuracy |
| -------- | -------- |
| Tyee     | 77.05    |
| Official | 76.01    |

Note: "Official" refers to the result from reproducing the original code.