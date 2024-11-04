import torch
import torch.nn as nn

class UpstreamDownstreamModel(nn.Module):
    def __init__(self, upstream, downstream, upstream_trainable):
        super().__init__()
        self.upstream = upstream
        self.downstream = downstream
        self.upstream_trainable = upstream_trainable

    def forward(self, wav, wav_len):
        with torch.set_grad_enabled(self.upstream_trainable):
            if not self.upstream_trainable:
                self.upstream.eval()
            h, h_len = self.upstream(wav, wav_len)
        return self.downstream(h,h_len)