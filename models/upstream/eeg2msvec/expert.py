#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : eeg2msvec_feat_extractor.py
@Time    : 2024/11/04 15:53:39
@Desc    : 
"""
import os
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from models.upstream.expert import BaseExpert 
from models.upstream.utils import merge_with_parent
from models.upstream.eeg2msvec.models.eeg2msvec import EEG2MSVec, EEG2MSVecConfig
from models.upstream.eeg2msvec.tasks.eeg_pretraining import EEGPretrainingConfig

# import sys
# sys.path.append("/home/taoz/code/PRL/models/")
# from upstream.feat_expert import UpstreamExpert 
# from upstream.utils import merge_with_parent
# from upstream.eeg2msvec.models.eeg2msvec import EEG2MSVec, EEG2MSVecConfig
# from upstream.eeg2msvec.tasks.eeg_pretraining import EEGPretrainingConfig


class EEG2MSVecExpert(BaseExpert):
    def __init__(
            self,
            ckpt: str,
            layer_hook_strs: list[str] = None,
            transforms: list[str] = None,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        model, task_cfg = self._build_model_from_fairseq_ckpt(ckpt)
        self.expert_model = model
        self.normalize = task_cfg.normalize
        self.np_normalize = False
        self.apply_padding_mask = False

        if transforms is None:
            transforms = [None for _ in layer_hook_strs]

        if isinstance(transforms, list):
            assert len(transforms) == len(layer_hook_strs)
            transforms = [eval(transform) for transform in transforms]
        else:
            transforms = [eval(transforms) for _ in layer_hook_strs]

        # add hooks
        if layer_hook_strs is not None:
            for layer_hook_str, transform in zip(layer_hook_strs, transforms):
                self.add_hook(
                    layer_hook_str, transform
                )

            def postprocess(xs):
                names, hiddens = zip(*xs)
                unpad_len = min([hidden.size(1) for hidden in hiddens])
                hiddens = [hidden[:, :unpad_len, :] for hidden in hiddens]
                return list(zip(names, hiddens))

            self.hook_postprocess = postprocess
        else:
            raise ValueError("layer_hook_strs is required.")

    def forward(self, wavs: list[torch.Tensor], padding_mask: torch.BoolTensor = None, *args, **kwargs) -> dict:
        # input data normal
        # device = wavs[0].device
        # if self.normalize:
        #     if self.np_normalize:
        #         tmp_wavs = [wav.cpu().numpy() for wav in wavs]
        #         wavs = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in tmp_wavs]
        #         wavs = [torch.from_numpy(wav).to(device) for wav in wavs]
        #     else:
        #         wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        # wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        # wav_padding_mask = ~torch.lt(
        #     torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
        #     wav_lengths.unsqueeze(1),
        # )
        # padded_wav = pad_sequence(wavs, batch_first=True)

        results = self.expert_model.extract_features(
            wavs, padding_mask
        )

        return results
        

    def get_downsample_rates(self, key: str) -> int:
        return 160
    
    def _build_model_from_fairseq_ckpt(self, ckpt_path: str, **override):
        """
        加载fairseq的模型
        :param str ckpt_path: 模型参数路径
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt_path {ckpt_path} not found.")

        ckpt_state = torch.load(ckpt_path, map_location="cpu")

        if "cfg" not in ckpt_state or "task" not in ckpt_state["cfg"]:
            raise ValueError(f"{ckpt_path} is not a valid checkpoint since the required key: cfg or task is missing")
        if "cfg" not in ckpt_state or "model" not in ckpt_state["cfg"]:
            raise ValueError(f"{ckpt_path} is not a valid checkpoint since the required key: cfg or model is missing")
        
        task_cfg = merge_with_parent(EEGPretrainingConfig, ckpt_state["cfg"]["task"])
        model_cfg = merge_with_parent(EEG2MSVecConfig, ckpt_state["cfg"]["model"])
        model = EEG2MSVec(model_cfg)
        model.load_state_dict(ckpt_state["model"])
        return model, task_cfg


if __name__ == "__main__":
    model = EEG2MSVecExpert(ckpt="/home/taoz/code/p4seq/eeg2msvec/2024-11-01/15-07-13/checkpoints/checkpoint_813_225000.pt")
    pass
