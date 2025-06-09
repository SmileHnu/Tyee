# Ninapro DB5 - sEMG Gesture Recognition

## 1. Experiment Overview

| Item                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Dataset**         | Ninapro Database 5 (Exercise 2)                              |
| **Signal Type**     | sEMG (surface Electromyography)                              |
| **Analysis Task**   | Gesture Recognition, a multi-class classification task to distinguish between 10 different hand postures from sEMG signals. |
| **Model Used**      | ResNet-18                                                    |
| **Reference Paper** | [EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html) |
| **Original Code**   | https://github.com/jehanyang/emgbench                        |

This experiment aims to use Tyee to perform an **sEMG gesture recognition** task on the **Ninapro DB5** dataset, following a complex workflow that includes **pre-training** and **personalized fine-tuning**. This page provides a detailed record of all the steps, configuration files, and expected results needed to reproduce the experiment.

------

## 2. Prerequisites

- **Download Location**: [NinaproDB5](https://ninapro.hevs.ch/instructions/DB5.html)

- Directory Structure: This experiment is conducted only on Exercise 2. Please download and decompress the dataset, arranging it according to the following structure:

  ```
  /path/to/data/NinaproDB5E2/
  ├── S1_E2_A.mat
  ├── S2_E2_A.mat
  └── ...
  ```

------

## 3. Model Configuration

### 3.1 Model Selection

This experiment uses the classic **ResNet-18** model. To process the one-dimensional sEMG signals, we first convert them into two-dimensional images (such as spectrograms or similar representations), which are then fed into the ResNet-18 model designed for image recognition. The model is loaded via the `timm` library.

### 3.2 Pre-trained Weights Information

In this experiment, the ResNet-18 backbone is first loaded with weights pre-trained on **ImageNet** and is then further updated during our **Phase 1 pre-training**.

- **Weight Source**: ImageNet (handled automatically by the `timm` library)
- **Usage**: Set `model.pretrained: true` in the configuration file.

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`.

### 4.1 Dataset Splitting

This experiment's splitting strategy is unique, following the method in the EMGBench paper. It is divided into two nested levels to simulate the scenario of adapting to a new, unseen user:

1. First-Level Split (Pre-training Phase - `LeaveOneOut`):
   - **Strategy**: Uses **Leave-One-Subject-Out** cross-validation. In each "fold," one subject is designated as the "left-out subject," and their data is temporarily set aside. The data from all other subjects is used as the training set for the pre-training phase.
2. Second-Level Split (Fine-tuning Phase - `HoldOutPerSubject`):
   - **Strategy**: The data from the subject left out in the first level is subjected to an internal hold-out split.
   - Split Ratios:
     - **20%** of the data is used as the **fine-tuning training set**.
     - **40%** of the data is used as the **validation set**.
     - **40%** of the data is used as the **test set**.
   - All splits are **stratified** by gesture class (`stratify: gesture`) to ensure a balanced data distribution.

### 4.2 Data Processing Pipeline

The data preprocessing pipeline, defined in the configuration file, has the core step of converting 1D sEMG time-series signals into 2D images:

1. **Signal Filtering (`Filter`)**: Applies a 3rd-order Butterworth band-pass filter to the raw sEMG signal.
2. **Signal Reshaping (`Reshape`)**: Reshapes the filtered signal segment into a `16x50` 2D matrix.
3. **Image Conversion (`ToImage`)**: Converts the 2D matrix into an image using the `viridis` color map.
4. **Resizing (`ImageResize`)**: Online, the generated image is resized to `224x224` to match the standard input size for ResNet-18.

### 4.3 Task Definition

- **Task Type**: `ninapro_db5_task.NinaproDB5Task`
- **Core Logic**: This task is responsible for receiving the processed sEMG image `emg` and the gesture label `gesture`, performing a forward pass through the ResNet-18 model, and computing the loss using **CrossEntropyLoss** to drive model training.

### 4.4 Training Strategy

- **Optimizer**: Uses **Adam** with a learning rate of **5e-4**.
- **Training Periods**: The pre-training phase runs for **100 epochs**, and the fine-tuning phase runs for **750 epochs**.
- **Evaluation Metric**: Uses **accuracy** as the core evaluation metric to select the best model.

### 4.5 Full Configuration File

```yaml
common:
  seed: 0
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 64
  dataset: ninapro_db5_dataset.NinaproDB5Dataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/NinaproDB5E2'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_ninapro_db5/train"
  io_mode: hdf5
  split: 
  	# Configuration for the pretrain phase
    # select: LeaveOneOutAndHoldOutET
    # init_params:
    #   split_path: /mnt/ssd/lingyus/tyee_ninapro_db5/split_pretrain
    #   stratify: gesture
    #   shuffle: false
    #   test_size: 0.4
    #   group_by: subject_id
    
    # Configuration for the fine-tuning phase
    select: HoldOutPerSubjectET
    init_params:
      split_path: /mnt/ssd/lingyus/tyee_ninapro_db5/split_finetune
      stratify: gesture
      shuffle: false
      val_size: 0.4
      test_size: 0.4
    run_params:
      subject: 3 # Example: fine-tuning and testing on subject 3

  offline_label_transform:
    - select: Mapping
      mapping: {0: 0, 17: 1, 18: 2, 20: 3, 21: 4, 22: 5, 25: 6, 26: 7, 27: 8, 28: 9}
      source: gesture
      target: gesture
    - select: OneHotEncode
      num: 10
      source: gesture
      target: gesture

  offline_signal_transform:
    - select: Filter
      l_freq: 5.0
      method: iir
      iir_params: {order: 3, ftype: butter, padlen: 12}
      phase: zero
      source: emg
      target: emg
    - select: Reshape
      shape: 800
      source: emg
      target: emg
    - select: ToImage
      length: 16
      width: 50
      cmap: viridis
      source: emg
      target: emg
    - select: ToNumpyFloat16
      source: emg
      target: emg
  
  online_signal_transform:
    - select: ImageResize
      size: [224, 224]
      source: emg
      target: emg

model:
  select: resnet18.resnet18
  num_classes: 10
  # ImageNet pre-trained weights
  pretrained: true
  
  # Fine-tuning phase loads the model pre-trained in Phase 1
  checkpoint_path: /home/lingyus/code/PRL/experiments/2025-05-23/11-24-05-ninapro_db5_task.NinaproDB5Task/checkpoint/fold_3/checkpoint_best.pt

optimizer:
  lr:  5e-4
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: ninapro_db5_task.NinaproDB5Task

trainer:
  fp16: false
  total_epochs: 750
  update_interval: 1
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, balanced_accuracy]
```

------

## 5. Replication Steps

This experiment consists of two phases:

1. **Phase 1: Pre-training**

   - First, you need to configure and run a pre-training process. This typically involves modifying the `split` section of the `config.yaml` file to use a `LeaveOneOut`-like strategy, training on all subjects except for the one left out.
   - Save the trained model checkpoint for each left-out subject.

2. **Phase 2: Personalized Fine-tuning and Evaluation**

   - **Confirm Configuration File**: Use the `config.yaml` file provided above. Ensure that all data paths (`root_path`, `io_path`, `split_path`) and the path to the model pre-trained in Phase 1 (`model.checkpoint_path`) are set correctly.

   - Run the Experiment: From the project's root directory, execute the following command. This command will load the pre-trained model and perform fine-tuning and evaluation on the data of the left-out subject (e.g., S3).

     ```bash
     python main.py --config config/ninapro_db5_finetune.yaml
     ```

     (Please replace `config/ninapro_db5_finetune.yaml` with your actual configuration file path.)

   - **Check the Results**: All experiment outputs will be saved in the directory specified by `common.exp_dir`.

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain test results similar to the table below. As this experiment uses leave-one-subject-out cross-validation, the data in the table represents the **average test performance across all subjects (folds)**. In each fold, the best model is selected based on the **best accuracy** achieved on its validation set and is then finally evaluated on the corresponding test set.

|          | Accuracy |
| -------- | -------- |
| Tyee     | 71.1     |
| Official | 68.3     |