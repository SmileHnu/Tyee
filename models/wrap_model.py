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
        with torch.set_grad_enabled(self.upstream_trainable):
            if not self.upstream_trainable:
                self.upstream.eval()
            h, h_len = self.upstream(wav,  *args, **kwargs)
        return self.downstream(h, h_len)