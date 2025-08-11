# PhysioNet P300 - P300 Potential Detection

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | PhysioP300                                                   |
| **Signal Type**     | EEG                                                          |
| **Analysis Task**   | P300 Speller Paradigm, a binary classification task to determine if an EEG signal contains a P300 event-related potential. |
| **Model Used**      | EEGPT                                                        |
| **Reference Paper** | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| **Original Code**   | https://github.com/BINE022/EEGPT                             |

This experiment aims to use Tyee to perform a **P300 potential detection** task on the **PhysioNet P300** dataset by applying **Linear Probing** to the **EEGPT** pre-trained model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for linear evaluation of large pre-trained models in Tyee.

------

## 2. Prerequisites

- **Download Location**: [PhysioP300](https://physionet.org/content/erpbci/1.0.0/)

- **Directory Structure**: Please download and decompress the dataset and arrange it according to the following structure:

  ```
  /path/to/data/erp-based-brain-computer-interface-recordings-1.0.0/files/
  ├── s01/
  │   └── rc01.edf
  │   └── ...
  ├── s02/
  │   └── rc01.edf
  │   └── ...
  └── ...
  ```

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the **EEGPT** model. We employ a **Linear Probing** strategy, which means all parameters of the EEGPT backbone network are **frozen** during training. Only a newly added learnable channel-scaling layer (`chan_scale`) and two linear layers (`linear_probe1`, `linear_probe2`) are trained.

### 3.2 Pre-trained Weights Information

In this experiment, we load the **official upstream pre-trained weights** for **EEGPT** (`eegpt_mcae_58chs_4s_large4E.ckpt`) as the model's starting point.

- **Weight Source**: The pre-trained weights are provided by the original model authors.

- **Download Link**: https://figshare.com/s/e37df4f8a907a866df4b
- **Usage**: Please place the downloaded weight file at the path specified by `model.load_path` in the configuration file. The model will automatically load these weights upon initialization.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Special Note**: The original dataset contains 12 subjects. However, to align with the experimental setup of the original paper, this experiment **excludes subjects S8, S10, and S12**, and is conducted only on the data from the remaining 9 subjects.
- **Splitting Strategy (`KFoldCross`)**: A **9-Fold Cross-Validation** is applied to the remaining 9 subjects. The data is grouped by subject ID (`group_by: subject_id`) before splitting, which is effectively equivalent to Leave-One-Subject-Out cross-validation.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is fully defined in the `offline_signal_transform` section of the configuration file. The core steps include:

1. **Channel Selection (`PickChannels`)**: Selects 64 standard EEG channels.
2. **Baseline Correction (`Baseline`)**: Uses the first `1435` time points of each trial as a baseline and subtracts its mean from the entire trial.
3. **Filtering (`Filter`)**: Applies a 0-120 Hz band-pass filter.
4. **Resampling (`Resample`)**: Resamples the signal to a uniform sampling rate of 256 Hz.
5. **Scaling (`Scale`)**: Multiplies the signal values by `1e-3`.

### 4.3 Task Definition

- **Task Type**: `physiop300_task.PhysioP300Task`
- **Core Logic**:
  - **Optimizer Parameter Setup (`set_optimizer_params`)**: This method is customized to **only return the parameters of the newly added `chan_scale`, `linear_probe1`, and `linear_probe2` layers** for optimization, thereby freezing the EEGPT backbone.
  - **Train/Validation Step**: Receives the EEG signal `x` and label `label`, gets the prediction `pred` from the model, and computes the loss using **CrossEntropyLoss**.

### 4.4 Training Strategy

- **Optimizer**: Uses **AdamW** with a weight decay of `0.01`.
- **Learning Rate Scheduler (`OneCycleScheduler`)**: Employs the **One-Cycle LR** policy. Over 100 epochs, the learning rate first increases linearly to a maximum of `4e-4` (within the first 20% of cycles) and then anneals down following a cosine schedule.
- **Training Period**: Trains for a total of **100 epochs**.
- **Evaluation Metric**: Uses **balanced accuracy** as the core evaluation metric to select the best model, while also calculating a variety of detailed metrics including `roc_auc`.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 7
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: physiop300_dataset.PhysioP300Dataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/erp-based-brain-computer-interface-recordings-1.0.0'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_physiop300/train"
  io_mode: hdf5
  io_chunks: 512
  include_end: true
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_physiop300/split
      group_by: subject_id
      n_splits: 9
      shuffle: false
  
  offline_signal_transform:
    - select: Compose
      transforms:
        - select: PickChannels
          channels: ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'IZ', 'OZ', 'POZ', 'PZ', 'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
        - select: Baseline
          baseline_end: 1435
          axis: 1
        - select: Filter
          l_freq: 0
          h_freq: 120
          method: iir
        - select: Resample
          desired_freq: 256
          pad: edge
        - select: Scale
          scale_factor: 1e-3
      source: eeg
      target: eeg

lr_scheduler:
  select: OneCycleScheduler
  max_lr: 4e-4
  epochs: 100
  pct_start: 0.2

model:
  select: eegpt.linear_probe_EEGPT_PhysioP300.LitEEGPTCausal
  load_path: /home/lingyus/code/PRL/models/eegpt/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt

optimizer:
  lr: 4e-4
  select: AdamW
  weight_decay: 0.01

task:
  loss:
    select: CrossEntropyLoss
  select: physiop300_task.PhysioP300Task

trainer:
  fp16: true
  total_epochs: 100
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: balanced_accuracy
    mode: max
  metrics: [balanced_accuracy, accuracy, f1_weighted, cohen_kappa, roc_auc]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) and the pre-trained model path (`model.load_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/physiop300.yaml
   ```

   (Please replace `config/physiop300.yaml` with your actual configuration file path.)

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. As this experiment uses 9-fold cross-validation, the table shows the **average performance of the best results from the validation set across all folds**.

|          | Balanced Accuracy | Cohen Kappa | ROC AUC |
| -------- | ----------------- | ----------- | ------- |
| Tyee     | 66.51             | 37.74       | 81.16   |
| Official | 65.02             | 29.99       | 71.68   |

### Result Analysis

The Tyee implementation shows notably better performance, particularly in ROC AUC (+9.48), compared to the official results. This improvement is primarily attributed to a critical preprocessing logic error discovered in the original codebase.

**Preprocessing Issue in Official Code**: 

The original preprocessing code contains a significant bug that causes sample overwriting and loss:

```python
spath = dataset_fold+f'{y}/'
os.makedirs(path,exist_ok=True)
spath = spath + f'{i}.sub{sub}'
torch.save(x, spath)
```

As shown in the code above, different files from the same subject generate identical sample names, causing later samples to overwrite earlier ones. This results in substantial sample loss during preprocessing.

**Verification Experiments**:

To validate this issue, we corrected the preprocessing logic and tested both versions on the original codebase:

| Version | Balanced Accuracy | Cohen Kappa | ROC AUC |
|---------|------------------|-------------|---------|
| Original (Buggy) | 71.62 | 42.91 | 77.71 |
| Corrected | 67.11 | 37.24 | 80.12 |

Both results differ from the official reported results. The original buggy version shows inflated performance due to reduced sample size (fewer samples available for training/testing), while the corrected version provides more reliable results with the complete dataset.
