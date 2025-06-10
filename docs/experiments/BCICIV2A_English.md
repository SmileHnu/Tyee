# BCI Competition IV Dataset 2a - Motor Imagery

## 1. Experiment Overview

| Item               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| **Dataset**        | BCI Competition IV Dataset 2a (BCICIV 2a)                    |
| **Signal Type**    | EEG                                                          |
| **Analysis Task**  | Motor Imagery Classification, 4-class task                   |
| **Model Used**     | Conformer                                                    |
| **Original Paper** | [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/abstract/document/9991178/) |
| **Original Code**  | https://github.com/eeyhsong/EEG-Conformer                    |
| **Special Note**   | This experiment is conducted on a single-subject basis. The following description uses subject A09 as an example. |

This experiment demonstrates how to use Tyee to perform a 4-class motor imagery classification task on the **BCICIV 2a** dataset using the **Conformer** model. This page details all the necessary steps, configuration files, and expected results to reproduce the experiment, serving as a practical guide for motor imagery analysis with Tyee.

------

## 2. Prerequisites

### 2.1 Data Download

The BCICIV 2a dataset provides signal data and labels separately, requiring two separate downloads:

- **Signal Data Download**: [BCICIV_2a_gdf.zip](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip)
  - Contains EEG signal data for all subjects (.gdf format)
  - Includes both training and test set signals, but the test set in .gdf files **has no labels**

- **Complete Labels Download**: [true_labels.zip](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)
  - Contains complete labels for both training and test sets (.mat format)
  - This is the official version that includes the true labels for the test set

### 2.2 Directory Structure

Since this experiment models each subject individually, please create a separate folder for each subject and place their corresponding training (`T` files) and evaluation (`E` files) from both downloads inside it. For subject `A09`, the directory structure should be as follows:

```
/path/to/data/BCICIV_2a/
└── A09/
    ├── A09T.gdf          # Training set signal data
    ├── A09T.mat          # Training set labels
    ├── A09E.gdf          # Test set signal data
    └── A09E.mat          # Test set labels
```

**Important Notes**:
- `.gdf` files come from `BCICIV_2a_gdf.zip`
- `.mat` files come from `true_labels.zip`
- Ensure that signal files and label files have matching subject IDs

------

## 3. Model Configuration

This experiment uses the **Conformer** model for motor imagery classification. Its key parameters are set in the configuration file as follows:

- **`n_outputs`**: 4 (corresponding to the 4 motor imagery classes)
- **`n_chans`**: 22 (corresponding to the 22 EEG channels)
- **`n_times`**: 1000 (the number of time points per sample)

------

## 4. Experiment Configuration & Data Processing

All settings for this experiment are centrally managed by a single configuration file, `config.yaml`. We use **Adam** as the optimizer with an initial learning rate of **0.0002** and the standard **CrossEntropyLoss**. The training is conducted for a total of **2000 epochs**.

### 4.1 Dataset Splitting

This experiment strictly follows the official splitting method of the BCICIV 2a dataset:

- **Splitting Strategy**: The official training set (`T` file, corresponding to `session_id=0`) is used for model training, and the evaluation set (`E` file, corresponding to `session_id=1`) is used as the validation set to evaluate model performance.
- **Configuration Implementation**: In the configuration file, this split is achieved through the `HoldOutCross` strategy by setting `group_by: session_id` and `val_size: 0.5`.

### 4.2 Data Processing Pipeline

We apply a two-stage processing pipeline to the data: **Offline** and **Online**.

1. Offline Processing:
   - **Bandpass Filtering (`Cheby2Filter`)**: Applies a 4-40 Hz bandpass filter to the EEG signals.
2. Online Processing:
   - **Z-Score Normalization (`ZScoreNormalize`)**: Uses the global mean and standard deviation from the subject's **training set** to normalize the data. This set of `mean` and `std` values is pre-calculated for each subject. See the **Appendix** for the calculation method.

### 4.3 Task Definition

- **Task Type**: `bciciv2a_task.BCICIV2aTask`
- **Core Logic**: This task is responsible for receiving the model's predictions and data labels, then computing the loss value using **CrossEntropyLoss** to drive the model training.

### 4.4 Full Configuration File

The following is the complete configuration used for this experiment (example for subject A09):

```yaml
common:
  seed: 2021
  verbose: true
  exp_dir: experiments

dataset:
  batch_size: 72
  dataset: bciciv2a_dataset.BCICIV2ADataset
  num_workers: 4
  root_path:
    train: '/mnt/ssd/lingyus/BCICIV_2a/A09'
  io_path:
    train: "/mnt/ssd/lingyus/tyee_bciciv2a/A09"
  io_mode: hdf5
  split: 
    select: HoldOutCross
    init_params:
      split_path: '/mnt/ssd/lingyus/tyee_bciciv2a/split/A09'
      group_by: session_id
      val_size: 0.5
      random_state: 4523

  offline_signal_transform:
    - select: Cheby2Filter
      l_freq: 4
      h_freq: 40
      source: eeg
      target: eeg
    - select: Select
      key: ['eeg']

  online_signal_transform:
    - select: ZScoreNormalize
      mean: -0.000831605672567208
      std: 9.915488018511994
      source: eeg
      target: eeg

model:
  select: conformer.Conformer
  n_outputs: 4
  n_chans: 22
  n_times: 1000

optimizer:
  lr: 0.0002
  select: Adam

task:
  loss:
    select: CrossEntropyLoss
  select: bciciv2a_task.BCICIV2aTask

trainer:
  fp16: true
  total_epochs: 2000
  log_interval: 20
  eval_metric:
    select: accuracy
    mode: max
  metrics: [accuracy, cohen_kappa]
```

------

## 5. Replication Steps

1. **Confirm Configuration File**: Ensure that the data paths (`root_path`, `io_path`, `split_path`) in the `config.yaml` file have been updated to your actual storage locations.

2. **Run the Experiment**: From the project's root directory, execute the following command.

   ```bash
   python main.py --config config/bciciv2a.yaml
   ```

   *(Please replace `main.py` and the config file path with your actual execution script and path.)*

3. **Check the Results**: All experiment outputs, including **training logs**, **model weights (.pt files)**, and **TensorBoard results**, will be saved in the directory specified by `common.exp_dir` in the configuration file, within a timestamped subfolder (e.g., `./experiments/2025-06-07/10-30-00/`).

------

## 6. Expected Results

After successfully reproducing this experiment, you should be able to obtain validation results similar to the table below. The table shows the **best performance** achieved by the model on the validation set during the entire training process.

| Subject             | Accuracy  |
| ------------------- | --------- |
| 1                   | 85.07     |
| 2                   | 59.03     |
| 3                   | 91.67     |
| 4                   | 77.08     |
| 5                   | 48.26     |
| 6                   | 59.03     |
| 7                   | 89.58     |
| 8                   | 82.64     |
| 9                   | 85.07     |
| **Tyee Average**    | **75.27** |
| **Official Result** | **74.91** |

Note: Data augmentation was used in the official experiment, but not in this experiment. The compared experimental results are also from a setup without data augmentation.

------

## Appendix: Calculating Global Mean and Standard Deviation for the Training Set

To ensure that no information from the validation and test sets leaks into the normalization process, we must use data **only from the training set** to calculate the global mean (`mean`) and standard deviation (`std`). These calculated values are then hard-coded into the `online_signal_transform` section of the configuration file to be applied to all data.

The following is an example script for calculating these statistics for a single subject (e.g., A09):

```python
import numpy as np
from tyee.dataset import BCICIV2ADataset # Assume this is your dataset class, please replace with the actual import path
from tyee.dataset.transform import Cheby2Filter, Select # Assume these are your offline transform classes

# 1. Define offline transforms consistent with the configuration file
offline_signal_transform = [
    Cheby2Filter(l_freq=4, h_freq=40, source='eeg', target='eeg'),
    Select(key=['eeg'])
]

# 2. Initialize the dataset instance; Tyee will automatically handle offline processing and caching
# Note: Do not apply the online ZScoreNormalize at this stage
dataset = BCICIV2ADataset(
    root_path='/mnt/ssd/lingyus/BCICIV_2a/A09',
    io_path='/mnt/ssd/lingyus/tyee_bciciv2a/A09',
    io_mode='hdf5',
    io_chunks=750,
    offline_signal_transform=offline_signal_transform
)

# 3. Filter the indices for the training set samples based on session_id
# In the BCICIV 2a dataset, 'T' files usually correspond to session 0, and 'E' files to session 1
train_indices = dataset.info[dataset.info['session_id'] == 0].index.tolist()

# 4. Collect the EEG data for all training set samples
all_train_eeg = []
for idx in train_indices:
    eeg_data = dataset[idx]['eeg']  # shape: (channels, time_points)
    all_train_eeg.append(eeg_data)

# 5. Concatenate all data and calculate the global mean and standard deviation
# Concatenate along the time axis to compute overall statistics across all channels and time points
full_train_data = np.concatenate(all_train_eeg, axis=-1) 
mean = np.mean(full_train_data)
std = np.std(full_train_data)

print(f"Statistics for Subject A09:")
print(f"Mean: {mean}")
print(f"Std: {std}")
```

You need to run this script separately for each subject and fill in the resulting `mean` and `std` values into their respective configuration files.

