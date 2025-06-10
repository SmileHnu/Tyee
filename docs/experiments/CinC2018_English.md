# CinC2018 - Sleep Staging

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | PhysioNet Challenge 2018 (CinC2018)                          |
| **Signal Type**     | EEG, EOG, ECG, Chest Respiration, Abdominal Respiration, SaO2 |
| **Analysis Task**   | Sleep Staging, 5-class task (Wake, N1, N2, N3, REM)          |
| **Model Used**      | SleepFM                                                      |
| **Reference Paper** | [SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity,ECG and Respiratory Signals](https://arxiv.org/abs/2405.17766) |
| **Original Code**   | https://github.com/rthapa84/sleepfm-codebase                 |

This experiment demonstrates how to use Tyee to perform a sleep staging task on the **CinC2018** dataset by fine-tuning the **SleepFM** multi-modal pre-trained model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for applying large multi-modal pre-trained models in Tyee.

------

## 2. Prerequisites

- **Download Location**: [CinC2018](https://physionet.org/content/challenge-2018/1.0.0/)

- Directory Structure: Please download and decompress the dataset, and then manually organize it into `train`, `valid`, and `test` subdirectories according to the splitting method described in [Section 4.1](#41-dataset-splitting). The final directory structure should be as follows:

  ```
  /path/to/data/challenge-2018-split/
  ├── train/
  │   ├── tr03-0001/
  │   └── ...
  ├── valid/
  │   ├── tr03-0002/
  │   └── ...
  └── test/
      ├── tr03-0004/
      └── ...
  ```

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the **SleepFM** model. SleepFM is a multi-modal foundational model pre-trained on massive amounts of sleep data via self-supervision. It includes three parallel `EffNet` encoders to process brain activity, cardiac, and respiratory signals respectively, making it highly suitable for complex sleep staging tasks.

### 3.2 Model Parameters

The key parameters for this model in this experiment are set in the configuration file as follows:

- **`num_classes`**: 5 (The number of classes for the downstream classification task, i.e., 5 sleep stages)
- **`bas_in_channels`**: 5 (The number of input channels for the brain activity (BAS) modality)
- **`ecg_in_channels`**: 1 (The number of input channels for the ECG modality)
- **`resp_in_channels`**: 3 (The number of input channels for the respiratory modality)
- **`embedding_dim`**: 512 (The embedding feature dimension output by each encoder)
- **`freeze_encoders`**: true (Freezes the parameters of the three upstream encoders, training only the downstream classification head)
- **`effnet_depth`**: `[1, 2, 2, 3, 3, 3, 3]` (The number of modules in each stage of the EffNet encoder)
- **`effnet_expansion`**: 6 (The expansion factor for the MBConv modules in EffNet)

### 3.3 Pre-trained Weights Information

In this experiment, we load the official upstream pre-trained weights for SleepFM to initialize the three encoders. Based on the `freeze_encoders: true` setting, their parameters are frozen, and only the downstream classification head is trained.

- **Weight Source**: The pre-trained weights are provided by the original model authors.
- **Download Link**: [SleepFM Checkpoints](https://github.com/rthapa84/sleepfm-codebase/tree/main/sleepfm/checkpoint)
- **Usage**: Please place the downloaded weight file at the path specified by `model.pretrained_checkpoint_path` in the configuration file. Tyee will automatically load these weights at the start of the experiment.

### 3.4 Downstream Task Classifier

The downstream task in the original SleepFM code uses an `sklearn` logistic regression classifier. Since Tyee is an end-to-end PyTorch framework, we **use a single-layer `nn.Linear`** in this experiment to replace the original logistic regression, serving as the classification head that connects the fused multi-modal features to the final classification result.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

- **Splitting Strategy**: We strictly follow the dataset splitting method from the original SleepFM paper's code. The original authors divided the entire dataset into a **pre-training set (75%), a training set, a validation set, and a test set**.
- **Implementation**: As this experiment only involves fine-tuning for a downstream task, the large pre-training set is not needed. We directly adopt the subject lists for the **training, validation, and test sets** as defined by the original authors (see [Appendix](#appendix-dataset-split-details) for details). You will need to manually extract the corresponding subject files from the original dataset and place them into `train`, `valid`, and `test` folders respectively. Consequently, we provide these three paths for both `root_path` and `io_path` in the configuration file and use the `NoSplit` strategy, indicating that the pre-split data should be used directly.

### 4.2 Data Processing Pipeline

We apply a multi-stage processing pipeline to the data:

1. Before-Segment Transform (`before_segment_transform`):
   - **Channel Selection/Concatenation (`PickChannels`, `Concat`)**: Selects 4 channels from the original multi-channel EEG and merges them with EOG to form the brain activity signal (ss); merges chest, abdominal respiration, and SaO2 signals into the respiratory signal (resp).
   - **Resampling (`Resample`)**: Resamples all modalities to a uniform sampling rate of 256 Hz.
2. Offline Transform(`offline_signal_transform`):
   - **Sliding Window (`SlideWindow`)**: Segments each record into individual 30-second windows.
3. Online Transform (`online_signal_transform`):
   - **Squeeze**: Removes extra dimensions to match the model's input shape.

### 4.3 Task Definition

- **Task Type**: `cinc2018_task.CinC2018Task`
- Core Logic:
  - **Optimizer Parameter Setup (`set_optimizer_params`)**: As the encoders are frozen, this method is customized to **only return the parameters of the downstream `nn.Linear` layer** for optimization.
  - **Train/Validation Step**: Receives the three modal inputs `ss`, `ecg`, and `resp`, gets the prediction `pred` from the SleepFM model, and computes the loss using **CrossEntropyLoss**.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **0.001** and a weight decay of `1e-4`.
- **Training Period**: Trains for a total of **100 epochs**.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model, while also calculating a variety of detailed metrics including `f1_macro` and `roc_auc_macro_ovr`.

### 4.5 Full Configuration File

The following is the complete configuration used for this experiment:

```yaml
common:
  seed: 42
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 128
  dataset: cinc2018_dataset.CinC2018Dataset
  num_workers: 16
  root_path:
    train: '/mnt/ssd/lingyus/challenge-2018-split/train'
    val: '/mnt/ssd/lingyus/challenge-2018-split/valid'
    test: '/mnt/ssd/lingyus/challenge-2018-split/test' 
  io_path:
    train: "/mnt/ssd/lingyus/tyee_cinc2018/train"
    val: "/mnt/ssd/lingyus/tyee_cinc2018/valid"
    test: "/mnt/ssd/lingyus/tyee_cinc2018/test"
  io_mode: hdf5
  io_chunks: 1
  split: 
    select: NoSplit
  
  before_segment_transform:
    - select: PickChannels
      channels: ["C3-M2", "C4-M1", "O1-M2", "O2-M1"]
      source: 'eeg'
      target: 'eeg'
    - select: Concat
      axis: 0
      source: ['eeg', 'eog']
      target: 'ss'
    - select: Concat
      axis: 0
      source: ['chest', 'sao2', 'abd']
      target: 'resp'
    - select: Resample
      desired_freq: 256
      source: 'ss'
      target: 'ss'
    - select: Resample
      desired_freq: 256
      source: 'resp'
      target: 'resp'
    - select: Resample
      desired_freq: 256
      source: 'ecg'
      target: 'ecg'
    - select: Select
      key: ['ss', 'resp', 'ecg']

  offline_signal_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'ss'
      target: 'ss'
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'resp'
      target: 'resp'
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'ecg'
      target: 'ecg'

  offline_label_transform:
    - select: SlideWindow
      window_size: 1
      stride: 1
      axis: 0
      source: 'stage'
      target: 'stage'
  online_signal_transform:
    - select: Squeeze
      axis: 0
      source: 'ss'
      target: 'ss'
    - select: Squeeze
      axis: 0
      source: 'resp'
      target: 'resp'
    - select: Squeeze
      axis: 0
      source: 'ecg'
      target: 'ecg'

  online_label_transform:
    - select: Squeeze
      axis: 0
      source: 'stage'
      target: 'stage'
  
model:
  select: sleepfm.sleepfm.SleepFM
  num_classes: 5
  bas_in_channels: 5
  ecg_in_channels: 1
  resp_in_channels: 3
  embedding_dim: 512
  freeze_encoders: true
  pretrained_checkpoint_path: "/home/lingyus/code/PRL/models/sleepfm/checkpoints/best.pt"
  effnet_depth: [1, 2, 2, 3, 3, 3, 3]
  effnet_channels_config: [32, 16, 24, 40, 80, 112, 192, 320, 1280]
  effnet_expansion: 6
  effnet_stride: 2
  effnet_dilation: 1

optimizer:
  lr:  0.001
  select: Adam
  weight_decay: 1e-4

task:
  loss:
    select: CrossEntropyLoss
  select: cinc2018_task.CinC2018Task

trainer:
  fp16: false
  total_epochs: 100
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, precision_macro, f1_macro, recall_macro, roc_auc_macro_ovr, pr_auc_macro,
            precision_weighted, f1_weighted, recall_weighted, roc_auc_weighted_ovr, pr_auc_weighted]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`) and the pre-trained model path (`model.pretrained_checkpoint_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/cinc2018.yaml
   ```

   *(Please replace `config/cinc2018.yaml` with your actual configuration file path.)*

3. **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`, within a timestamped subfolder.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. The evaluation methodology is as follows: the best model is trained on the training set, selected based on the **best accuracy** achieved on the validation set, and finally evaluated on the test set.

|               | AUROC (Macro) | AUPRC (Macro) | F1 (Macro) |
| ------------- | ------------- | ------------- | ---------- |
| Tyee          | 90.27         | 71.08         | 63.9       |
| Official Code | 90.14         | 70.35         | 64.7       |

------

## Appendix: Dataset Split Details

The following are the subject lists for the training, validation, and test sets used in this experiment, consistent with the official SleepFM codebase.

### Training Set

```python
['tr07-0235', 'tr03-0167', 'tr03-0428', 'tr05-0707', 'tr04-0583', 'tr07-0291', 'tr11-0016', 'tr11-0640', 'tr04-1064', 'tr12-0339', 'tr11-0050', 'tr14-0291', 'tr04-1021', 'tr09-0593', 'tr07-0281', 'tr07-0752', 'tr09-0331', 'tr08-0295', 'tr12-0395', 'tr05-0572', 'tr03-0678', 'tr10-0094', 'tr10-0707', 'tr06-0379', 'tr07-0125', 'tr05-0028', 'tr06-0302', 'tr03-0212', 'tr07-0796', 'tr07-0542', 'tr05-1190', 'tr03-0413', 'tr12-0414', 'tr03-0052', 'tr12-0253', 'tr10-0363', 'tr07-0162', 'tr11-0029', 'tr12-0672', 'tr03-0314', 'tr07-0709', 'tr03-1183', 'tr04-0231', 'tr10-0752', 'tr06-0313', 'tr06-0390', 'tr03-0933', 'tr04-0695', 'tr09-0423', 'tr04-0265', 'tr05-1675', 'tr11-0510', 'tr04-0041', 'tr05-1313', 'tr07-0343', 'tr12-0481', 'tr06-0865', 'tr13-0379', 'tr07-0874', 'tr05-0910', 'tr07-0579', 'tr07-0458', 'tr12-0497', 'tr04-0710', 'tr12-0607', 'tr04-1023', 'tr10-0477', 'tr12-0348', 'tr10-0869', 'tr10-0598', 'tr12-0560', 'tr03-0426', 'tr06-0764', 'tr11-0655', 'tr06-0850', 'tr07-0575', 'tr03-1115', 'tr11-0786', 'tr11-0659', 'tr13-0627', 'tr10-0336', 'tr11-0708', 'tr03-0257', 'tr11-0563', 'tr11-0006', 'tr06-0556', 'tr03-0697', 'tr13-0517', 'tr07-0043', 'tr03-0494', 'tr06-0122', 'tr04-0020', 'tr04-0029', 'tr10-0263', 'tr05-1375', 'tr12-0503', 'tr06-0014', 'tr05-1570', 'tr06-0771', 'tr12-0646', 'tr12-0209', 'tr12-0492', 'tr07-0681', 'tr04-1078', 'tr03-0773', 'tr03-0921', 'tr07-0056', 'tr05-0334', 'tr11-0767', 'tr13-0226', 'tr05-1558', 'tr03-0146', 'tr12-0173', 'tr05-0857', 'tr13-0589', 'tr07-0212', 'tr05-1176', 'tr06-0773', 'tr05-1128', 'tr13-0801', 'tr12-0681', 'tr14-0011', 'tr11-0080', 'tr09-0070', 'tr03-0394', 'tr05-0864', 'tr12-0319', 'tr07-0593', 'tr06-0084', 'tr11-0452', 'tr05-1653', 'tr11-0644', 'tr12-0321', 'tr04-0209']

```

### Validation Set

```python
['tr04-1097', 'tr04-0569', 'tr04-0121', 'tr11-0335', 'tr06-1117', 'tr12-0121', 'tr07-0023', 'tr14-0016', 'tr11-0573', 'tr08-0353', 'tr12-0097', 'tr12-0061', 'tr12-0425', 'tr12-0015', 'tr08-0012']

```

### Test Set

```python
['tr04-0939', 'tr11-0338', 'tr03-1160', 'tr09-0568', 'tr05-0326', 'tr04-0227', 'tr14-0064', 'tr07-0153', 'tr05-0226', 'tr05-0635', 'tr10-0853', 'tr13-0525', 'tr13-0505', 'tr04-1105', 'tr06-0447', 'tr10-0059', 'tr10-0511', 'tr04-0210', 'tr03-0100', 'tr13-0576', 'tr06-0705', 'tr12-0106', 'tr09-0575', 'tr04-0829', 'tr13-0685', 'tr09-0051', 'tr14-0110', 'tr03-1143', 'tr06-0883', 'tr05-1034', 'tr07-0568', 'tr03-0907', 'tr04-0208', 'tr03-0743', 'tr05-1404', 'tr12-0520', 'tr07-0601', 'tr12-0364', 'tr04-1096', 'tr05-0119', 'tr11-0457', 'tr07-0564', 'tr06-0609', 'tr09-0453', 'tr04-0699', 'tr04-0362', 'tr05-0443', 'tr04-0144', 'tr06-0404', 'tr10-0704', 'tr05-1377', 'tr04-0287', 'tr12-0003', 'tr04-0959', 'tr08-0256', 'tr05-1385', 'tr14-0268', 'tr05-0074', 'tr03-0198', 'tr03-0904', 'tr05-0174', 'tr07-0123', 'tr06-0242', 'tr05-0348', 'tr05-0646', 'tr03-0793', 'tr06-0721', 'tr13-0425', 'tr03-0802', 'tr07-0770', 'tr06-0644', 'tr03-0300', 'tr04-0568', 'tr11-0587', 'tr04-0014', 'tr05-1489', 'tr13-0374', 'tr04-0808', 'tr07-0168', 'tr03-1056', 'tr12-0426', 'tr04-0008', 'tr04-0785', 'tr04-0570', 'tr04-0117', 'tr14-0003', 'tr14-0240', 'tr05-0048', 'tr13-0646', 'tr09-0489', 'tr05-0301', 'tr03-0982', 'tr08-0183', 'tr05-0664', 'tr08-0315', 'tr12-0448', 'tr07-0566', 'tr05-0784', 'tr03-1292', 'tr05-0880']
```