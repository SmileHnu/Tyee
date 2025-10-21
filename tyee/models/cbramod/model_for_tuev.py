import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, use_pretrained_weights, cuda, foundation_dir, classifier, num_of_classes, dropout=0.1, *args, **kwargs):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if use_pretrained_weights:
            map_location = torch.device(f'cuda:{cuda}')
            self.backbone.load_state_dict(torch.load(foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        if classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 1 * 200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 1 * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 1 * 200, 1 * 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(1 * 200, 200),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out

