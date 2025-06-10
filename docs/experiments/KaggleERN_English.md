# Kaggle ERN - Event-Related Negativity Detection

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | KaggleERN                                                    |
| **Signal Type**     | EEG                                                          |
| **Analysis Task**   | Event-Related Negativity (ERN) Detection, a binary classification task to determine if a subject exhibits an error-related negativity response to a stimulus. |
| **Model Used**      | EEGPT                                                        |
| **Reference Paper** | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| **Original Code**   | https://github.com/BINE022/EEGPT                             |

This experiment demonstrates how to use Tyee to perform an ERN detection task on the **Kaggle ERN** dataset via **Linear Probing** of the **EEGPT** pre-trained model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for linear evaluation of large pre-trained models in Tyee.

------

## 2. Prerequisites

### 2.1 Data Download

- **Download Location**: [KaggleERN](https://www.kaggle.com/c/inria-bci-challenge/data)

### 2.2 Directory Structure

Please download and decompress the dataset and arrange it according to the following structure. **Note**: The original downloaded `TrainLabels.csv` and `true_labels.csv` files are usually placed at the same level as the `train/` and `test/` folders, and need to be manually moved to their corresponding folders:

```
/path/to/data/KaggleERN/
├── train/
│   ├── Data_S02_Sess01.csv
│   ├── ...
│   └── TrainLabels.csv          # Move to train folder
└── test/
    ├── Data_S01_Sess01.csv
    ├── ...
    └── true_labels.csv          # Move to test folder
```

**Important Notes**:
- After downloading, the label files `TrainLabels.csv` and `true_labels.csv` may be located in the dataset root directory
- Please ensure that `TrainLabels.csv` is moved to the `train/` folder
- Please ensure that `true_labels.csv` is moved to the `test/` folder
- This structure facilitates proper data and label reading by the Tyee framework

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the **EEGPT** model. It employs a **Linear Probing** strategy, which means all parameters of the EEGPT backbone network are **frozen** during training, and only the newly added linear layers are trained.

### 3.2 Pre-trained Weights Information

In this experiment, we load the **official upstream pre-trained weights** for **EEGPT** (`eegpt_mcae_58chs_4s_large4E.ckpt`) as the model's starting point.

- **Weight Source**: The pre-trained weights are provided by the original model authors.

- **Download Link**: https://figshare.com/s/e37df4f8a907a866df4b
- **Usage**: Please place the downloaded weight file at the path specified by `model.load_path` in the configuration file. The model will automatically load these weights upon initialization.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy (`KFoldCross`)**: This experiment uses **4-Fold Cross-Validation**. The data is grouped by subject ID (`group_by: subject_id`) before splitting to ensure that all data from a single subject belongs to the same fold, preventing data leakage.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline for this experiment is fully defined in the `offline_signal_transform` section of the configuration file. The core steps include:

1. **Normalization and Scaling (`MinMaxNormalize`, `Offset`, `Scale`)**: The signal is first normalized using Min-Max scaling, then adjusted via an offset and scaling factor to fit the range `[-1, 1]`.
2. **Channel Selection (`PickChannels`)**: Selects 19 standard EEG channels.

### 4.3 Task Definition

- **Task Type**: `kaggleern_task.KaggleERNTask`
- Core Logic:
  - **Optimizer Parameter Setup (`set_optimizer_params`)**: This method is customized to **only return the parameters of the newly added `chan_conv`, `linear_probe1`, and `linear_probe2` layers** for optimization, thereby freezing the EEGPT backbone.
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
  dataset: kaggleern_dataset.KaggleERNDataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/KaggleERN/train'
    test: '/mnt/ssd/lingyus/KaggleERN/test'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_kaggleern/train"
    test: "/mnt/ssd/lingyus/tyee_kaggleern/test"
  io_mode: hdf5
  io_chunks: 400
  split: 
    select: KFoldCross
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_kaggleern/split
      group_by: subject_id
      n_splits: 4
      shuffle: false
  
  offline_signal_transform:
    - select: MinMaxNormalize
      source: eeg
      target: eeg
    - select: Offset
      offset: -0.5
      source: eeg
      target: eeg
    - select: Scale
      scale_factor: 2.0
      source: eeg
      target: eeg
    - select: PickChannels
      channels: ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2']
      source: eeg
      target: eeg

lr_scheduler:
  select: OneCycleScheduler
  max_lr: 4e-4
  epochs: 100
  pct_start: 0.2

model:
  select: eegpt.linear_probe_EEGPT_KaggleERN.LitEEGPTCausal
  load_path: /home/lingyus/code/PRL/models/eegpt/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt

optimizer:
  lr: 4e-4
  select: AdamW
  weight_decay: 0.01

task:
  loss:
    select: CrossEntropyLoss
  select: kaggleern_task.KaggleERNTask

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
   python main.py --config config/kaggleern.yaml
   ```

   *(Please replace `config/kaggleern.yaml` with your actual configuration file path.)*

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. As this experiment uses 4-fold cross-validation, the table shows the **average performance of the test results across all folds**.

|          | Balanced Accuracy | Cohen Kappa | ROC AUC |
| -------- | ----------------- | ----------- | ------- |
| Tyee     | 61.17             | 22.68       | 67.15   |
| Official | 58.37             | 18.82       | 66.21   |