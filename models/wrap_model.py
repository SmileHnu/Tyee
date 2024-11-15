#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn


class WrappedMode(nn.Module):
    def __init__(
            self,
            upstream: nn.Module,
            downstream: nn.Module,
            upstream_trainable: bool = False
        ) -> None:
        super().__init__()
        self.upstream = upstream
        self.downstream = downstream
        self.upstream_trainable = upstream_trainable

    def forward(self, wav, *args, **kwargs):
        if self.upstream_trainable:
            hiddens = self.upstream(wav, *args, **kwargs)
        else:
            with torch.no_grad():
                hiddens = self.upstream(wav, *args, **kwargs)
        # print(hiddens.shape)
        return self.downstream(hiddens, *args, **kwargs)
