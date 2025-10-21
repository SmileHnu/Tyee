<div align="center">

# Tyee

<!-- Language Switch with Badges -->
<p>
  <a href="./README.md">
    <img src="https://img.shields.io/badge/Language-English-blue?style=flat-square&logo=google-translate" alt="English">
  </a>
  <a href="./README_Chinese.md">
    <img src="https://img.shields.io/badge/语言-中文-red?style=flat-square&logo=google-translate" alt="中文">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License: CC BY-NC 4.0">
  </a>
</p>
<!-- Installation Badges with Links -->
<p>
  <a href="#using-conda">
    <img src="https://img.shields.io/badge/Install-Conda-green?style=flat-square&logo=anaconda" alt="Conda Installation">
  </a>
  <a href="#using-docker">
    <img src="https://img.shields.io/badge/Install-Docker-blue?style=flat-square&logo=docker" alt="Docker Installation">
  </a>
  <a href="#-quick-start">
    <img src="https://img.shields.io/badge/Quick-Start-yellow?style=flat-square&logo=lightning" alt="Quick Start">
  </a>
</p>

</div>

## Introduction

Tyee is a PyTorch-based toolkit for intelligent analysis of physiological signals.  It aims to empower researchers and developers to efficiently perform task analysis on multi-source heterogeneous physiological signals and is dedicated to simplifying the entire workflow from data preprocessing to model training and evaluation, providing an integrated solution. 

<div align="center">

![Tyee框架概览](./img/Tyee-tyee.jpg)

</div>

*Architecture of the Tyee toolkit. It features modular components and end-to-end configurability, supporting a diverse range of physiological signals and healthcare applications via unified data and task interfaces.*

Tyee is committed to streamlining the research and development process for physiological signal analysis, offering comprehensive support from data preprocessing to model training and evaluation.

## ✨ Features

Tyee possesses the following core features:

- **Unified Data Interface & Configurable Preprocessing**: Offers a unified data interface and flexibly configurable preprocessing pipelines for diverse signal modalities (e.g., EEG, ECG, EMG).
- **Modular and Extensible Architecture**: Employs a modular design, facilitating easy integration of components and rapid prototyping across different tasks. 
- **Fully-Integrated Support**: Provides comprehensive integrated support from preprocessing to training, promoting reproducible and scalable experiments. 
- **Broad Signal and Task Coverage**: As shown in the figure above, the framework's design allows it to be easily applied to various physiological signals and analytical tasks. 
- **Built with PyTorch**: Fully leverages the flexibility and powerful ecosystem of PyTorch for efficient model development and training.

## 🛠️ Framework Modules

Tyee's core architecture adopts a modular design (as shown in the figure below), ensuring that each component is highly independent and easily extensible, achieving plug-and-play flexibility. 

<div align="center">

<img src="./img/architecture.jpg" width="60%" alt="Framework Architecture">

</div>

*Tyee module structure diagram, illustrating core components such as configuration, trainer, task, entities (data, model, preprocessor, etc.), and their interactive relationships.* 

The following table details the main modules of Tyee and their functions.  Each module name links to its dedicated README document, allowing you to delve deeper into specific implementations and usage methods.

| Module Name             | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| [Tyee.dataset](./docs/module/dataset_English.md)            | Provides a unified and flexible data interface to handle heterogeneous data. |
| [Tyee.dataset.io](./docs/module/dataset_io_English.md)         | Implements a unified input/output (I/O) to store the processing results of various heterogeneous datasets. |
| [Tyee.dataset.transform](./docs/module/dataset_transform_English.md)  | Offers rich preprocessing methods for users to choose flexibly. |
| [Tyee.dataset.split](./docs/module/dataset_split_English.md)      | Provides rich data splitting methods for users to choose, enabling settings at different granularities. |
| [Tyee.model](./docs/module/model_English.md)              | Offers commonly used task analysis models for users to reproduce. |
| [Tyee.criterion](./docs/module/criterions_English.md)          | Extends the loss functions based on PyTorch.                 |
| [Tyee.metrics](./docs/module/metrics_English.md)            | Provides rich classification and regression evaluation metrics. |
| Tyee.optim              | Uses PyTorch optimizers.                                     |
| [Tyee.optim.lr_scheduler](./docs/module/lr_scheduler_English.md) | Re-encapsulates PyTorch learning rate schedulers and extends them. |
| [Tyee.task](./docs/module/task_English.md)               | Responsible for initializing all components of the entity layer and defining the training logic from forward propagation to loss calculation for tasks. |
| [Tyee.trainer](./docs/module/trainer_English.md)            | Responsible for the end-to-end training and validation process. |
| [Tyee.config](./docs/module/config_English.md)             | Provides centralized control over the entire framework through YAML files. |

## 🚀 Installation Guide
This guide provides two main paths to get started with Tyee: **Direct Execution** (for quick development) and **Package Installation** (for stable use and deployment). Both methods require setting up a Conda or Docker environment first.
### Environment Requirements: 

- Python 3.10 
- PyTorch 2.5.0 

### Step 1: Set Up Your Environment

First, prepare your Python environment using either Conda or Docker.

#### Using Conda

**1. Clone the Code Repository**: 

```bash
$ git clone https://github.com/SmileHnu/Tyee.git
$ cd Tyee
```

**2. Create and Activate Virtual Environment**:

```bash
$ conda env create -n tyee -f environment.yml
$ conda activate tyee
```

#### Using Docker

**1. Clone the Code Repository**: 

```bash
$ git clone https://github.com/SmileHnu/Tyee.git
$ cd Tyee
```

**2. Build and Run Docker Container**:

```bash
# Build Docker image
$ docker build -t tyee:latest .

# Run container
$ docker run -it --name tyee-dev \
  -v $(pwd):/workspace/Tyee \
  -p 6006:6006 \
  --workdir /workspace/Tyee \
  tyee:latest bash
```

### Step 2: Choose Your Execution Method
After setting up the environment, you can run experiments in two ways.

#### Method 1: Direct Execution (No Installation Required)
This is the fastest way to get started. It's ideal for development and debugging as it doesn't require installing the package.

Simply run the train.py script from the project's root directory (Tyee/).

```bash
# Make sure you are in the project root directory (e.g., Tyee/)
# and your conda environment is activated.

$ python train.py -c tyee/config/mit_bih.yaml
```

#### Method 2: Package Installation (Recommended)
This method installs tyee as a standard Python package, providing the tyee-train command-line tool that can be run from anywhere.

##### 1. Install Tyee in Editable Mode:
This command should be run from the project root directory.
```bash
$ pip install -e .
```
##### 2. Run an Experiment:
You can now use the tyee-train command.
```bash
# Note that you need to provide the full or relative path to the config file.
$ tyee-train -c /path/to/your/Tyee/tyee/config/mit_bih.yaml
```

## ⚡ Quick Start

The following demonstrates the basic usage of Tyee through a MIT-BIH arrhythmia detection experiment. For complete experimental configuration and detailed instructions, please refer to: [MIT-BIH Experiment Details](./docs/experiments/MITBIH_English.md)

### Using Conda Environment

```bash
# Activate environment and run complete experiment workflow
$ conda activate tyee
$ bash quick_start.sh
```

### Using Docker Container

```bash
# Enter container and run complete experiment workflow
$ docker exec -it tyee-dev bash
$ bash quick_start.sh
```

After the experiment completes, you can view the experimental results in the `experiments/` directory, including training logs, model files, and TensorBoard visualization files. Use the following command to start TensorBoard:

```bash
$ tensorboard --logdir ./experiments/ --port 6006
```

### More Experiments

To explore more dataset and model combinations, please refer to the [Performance Benchmarks and Application Examples](#🧪-performance-benchmarks-and-application-examples) section.

## 📊 Supported Datasets and Models

We support a number of datasets in Tyee to facilitate user use and extension. 

### Supported Datasets

The following table summarizes the supported datasets and the signals and tasks they involve: 

| Dataset                                                      | Signal Type                         | Task                 |
| ------------------------------------------------------------ | ----------------------------------- | -------------------- |
| [TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuev) | EEG                                 | Event Classification |
| [TUAB](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab) | EEG                                 | Abnormal Detection   |
| [BCICIV-2A](https://www.bbci.de/competition/iv/#datasets)    | EEG, EOG                            | Motor Imagery        |
| [BCICIV-4](https://www.bbci.de/competition/iv/#datasets)     | ECoG                                | Finger Movement      |
| [KaggleERN](https://www.kaggle.com/c/inria-bci-challenge/data) | EEG                                 | ERN Detection        |
| [PhysioP300](https://physionet.org/content/erpbci/1.0.0/)    | EEG                                 | P300 Recognition     |
| [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)        | ECG                                 | Arrhythmia Detection |
| [NinaproDB5](https://ninapro.hevs.ch/instructions/DB5.html)  | EMG                                 | Gesture Recognition   |
| [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | PPG, 3D Acc, HR                     | HR Estimation        |
| [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html)     | EEG, EOG                            | Emotion Recognition  |
| [DEAP](http://eecs.qmul.ac.uk/mmv/datasets/deap/)            | EEG, EOG, EMG, GSR, RESP, PPG, TEMP | Emotion Recognition  |
| [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/) | EEG, EOG, EMG, RESP, TEMP           | Sleep Staging        |
| [CinC2018](https://physionet.org/content/challenge-2018/1.0.0/) | EEG, EOG, EMG, RESP, SpO2,ECG       | Sleep Staging        |

### Supported Models

Tyee integrates a series of commonly used physiological signal analysis models and also supports user-defined and extended models.  The following table summarizes the supported models and their corresponding references for users to consult and use: 

| Model                                                        | Paper                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [LaBraM](https://github.com/935963004/LaBraM)                | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP) |
| [EEGPT](https://github.com/BINE022/EEGPT)                    | [EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) |
| [BIOT](https://github.com/ycq091044/BIOT)                    | [BIOT: Cross-data Biosignal Learning in the Wild](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html) |
| [EEGConformer](https://github.com/eeyhsong/EEG-Conformer)    | [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/abstract/document/9991178/) |
| [FingerFlex](https://github.com/Irautak/FingerFlex)          | [FingerFlex: Inferring Finger Trajectories from ECoG signals](https://arxiv.org/abs/2211.01960) |
| [ECGResNet34](https://github.com/lxdv/ecg-classification)    | [Diagnosis of Diseases by ECG Using Convolutional Neural Networks](https://www.hse.ru/en/edu/vkr/368722189) |
| [EMGBech-ResNet18](https://github.com/jehanyang/emgbench)    | [EMGBench: Benchmarking Out-of-Distribution Generalization and Adaptation for Electromyography](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html) |
| [G2G-ResNet18](https://github.com/Jinminbox/G2G)             | [Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://dl.acm.org/doi/abs/10.1145/3581783.3612074) |
| [BeliefPPG](https://github.com/eth-siplab/BeliefPPG)         | [BeliefPPG: Uncertainty-aware Heart Rate Estimation from PPG signals via Belief Propagation](https://proceedings.mlr.press/v216/bieri23a.html) |
| [MLSTM-FCN](https://github.com/titu1994/MLSTM-FCN)           | [Multivariate LSTM-FCNs for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200) |
| [SalientSleepNet](https://github.com/ziyujia/SalientSleepNet) | [SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://arxiv.org/abs/2105.13864) |
| [SleepFM](https://github.com/rthapa84/sleepfm-codebase)      | [SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity,ECG and Respiratory Signals](https://arxiv.org/abs/2405.17766) |

## 🧪 Performance Benchmarks and Application Examples

To fully validate the performance and generalization capabilities of Tyee, we have conducted extensive experiments on 13 public physiological signal datasets.  The following shows the main experimental results and detailed configurations. These examples can also serve as a quick start reference for using this framework. 

<div align="center">

![](./img/radar.jpg)

</div>

*Radar plots comparing Tyee (red) vs. baselines (blue) in (a) unimodal and (b) multimodal healthcare tasks. Tyee matches or outperforms the baselines on 12 out of 13 datasets, demonstrating reliable and consistent benchmark reproduction.*

The following table summarizes the 13 datasets covered in this benchmark test, the corresponding signal types, analysis tasks, and the models used.  Each entry provides a link to the detailed experimental configuration (README) and the official repository of the relevant model, making it convenient for users to reproduce and conduct in-depth research.

| Dataset                                                      | Signal Type              | Task                 | Model                                                        | Detailed Experiment Link |
| ------------------------------------------------------------ | ------------------------ | -------------------- | ------------------------------------------------------------ | ------------------------ |
| [TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuev) | EEG                      | Event Classification | [LaBraM](https://github.com/935963004/LaBraM)                | [Experiment Details](./docs/experiments/TUEV_English.md)       |
| [TUAB](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tuab) | EEG                      | Abnormal Detection   | [LaBraM](https://github.com/935963004/LaBraM)                | [Experiment Details](./docs/experiments/TUAB_English.md)       |
| [BCICIV-2A](https://www.bbci.de/competition/iv/#datasets)    | EEG                      | Motor Imagery        | [EEGConformer](https://github.com/eeyhsong/EEG-Conformer)    | [Experiment Details](./docs/experiments/BCICIV2A_English.md)       |
| [BCICIV-4](https://www.bbci.de/competition/iv/#datasets)     | ECoG                     | Finger Movement      | [FingerFlex](https://github.com/Irautak/FingerFlex)          | [Experiment Details](./docs/experiments/BCICIV4_English.md)       |
| [KaggleERN](https://www.kaggle.com/c/inria-bci-challenge/data) | EEG                      | ERN Detection        | [EEGPT](https://github.com/BINE022/EEGPT)                    | [Experiment Details](./docs/experiments/KaggleERN_English.md)       |
| [PhysioP300](https://physionet.org/content/erpbci/1.0.0/)    | EEG                      | P300 Recognition     | [EEGPT](https://github.com/BINE022/EEGPT)                    | [Experiment Details](./docs/experiments/PhysioP300_English.md)       |
| [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)        | ECG                      | Arrhythmia Detection | [ECGResNet34](https://github.com/lxdv/ecg-classification)    | [Experiment Details](./docs/experiments/MITBIH_English.md)       |
| [NinaproDB5](https://ninapro.hevs.ch/instructions/DB5.html)  | EMG                      | Gesture Recognition   | [EMGBench-ResNet18](https://github.com/jehanyang/emgbench)   | [Experiment Details](./docs/experiments/NinaproDB5_English.md)       |
| [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) | PPG, 3D Acc          | HR Estimation        | [BeliefPPG](https://github.com/eth-siplab/BeliefPPG)         | [Experiment Details](./docs/experiments/DaLiA_English.md)       |
| [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html)     | EEG, EOG                 | Emotion Recognition  | [G2G-ResNet18](https://github.com/Jinminbox/G2G)             | [Experiment Details](./docs/experiments/SEEDV_English.md)       |
| [DEAP](http://eecs.qmul.ac.uk/mmv/datasets/deap/)            | GSR, RESP, PPG, TEMP     | Emotion Recognition  | [MLSTM-FCN](https://github.com/athar70/MLSTM)                | [Experiment Details](./docs/experiments/DEAP_English.md)       |
| [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/) | EEG, EOG                 | Sleep Staging        | [SalientSleepNet](https://github.com/ziyujia/SalientSleepNet) | [Experiment Details](./docs/experiments/SleepEDFx_English.md)       |
| [CinC2018](https://physionet.org/content/challenge-2018/1.0.0/) | EEG, EOG, RESP, SpO2,ECG | Sleep Staging        | [SleepFM](https://github.com/rthapa84/sleepfm-codebase)      | [Experiment Details](./docs/experiments/CinC2018_English.md)       |

## 🤝 Contributing

We welcome contributions in any form, including but not limited to: 

- Reporting Bugs 
- Submitting Feature Requests 
- Code Contributions 
- Documentation Improvements 

## 🙏 References and Acknowledgements

The design and implementation of Tyee have been inspired by and benefited greatly from the following outstanding open-source projects, and we express our sincere gratitude: 

- [PyTorch](https://github.com/pytorch/pytorch) 
- [mne](https://github.com/mne-tools/mne-python) 
- [s3prl](https://github.com/s3prl/s3prl)
- [fairseq](https://github.com/facebookresearch/fairseq) 
- [torcheeg](https://github.com/torcheeg/torcheeg)



## 📝 License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

Please see the [`LICENSE`](./LICENSE) file for the complete license text.

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## 📖 Citation

If you use Tyee in your research, please consider citing it as follows:
```
@inproceedings{
zhou2025tyee,
title={Tyee: A Unified, Modular, and Fully-Integrated Configurable Toolkit for Intelligent Physiological Health Care},
author={Tao Zhou and Lingyu Shu and Zixing Zhang and Jing Han},
booktitle={ACM Multimedia 2025 Open Source Track},
year={2025},
url={https://openreview.net/forum?id=k4zQNEMJrL}
}
```