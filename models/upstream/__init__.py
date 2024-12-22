#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : __init__.py
@Time    : 2024/11/09 23:46:48
@Desc    : 
"""
from models.upstream.wavelet import Wavelet
from models.upstream.eeg2msvec.expert import EEG2MSVecExpert
from models.upstream.mlp import MLP
from models.upstream.labram.modeling_finetune import labram_base_patch200_200, labram_huge_patch200_200, labram_large_patch200_200
from models.upstream.eegpt.finetune_EEGPT_SleepEDF import LitEEGPTCausal as EEGPT_SleepEDF
from models.upstream.eegpt.linear_probe_EEGPT_BCIC2A import LitEEGPTCausal as EEGPT_BCICIV2A
from models.upstream.eegpt.linear_probe_EEGPT_KaggleERN import LitEEGPTCausal as EEGPT_KaggleERN
from models.upstream.eegpt.linear_probe_EEGPT_PhysioP300 import LitEEGPTCausal as EEGPT_PhysioP300