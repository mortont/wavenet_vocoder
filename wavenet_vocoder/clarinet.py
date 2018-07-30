# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .modules import Embedding
from .modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from .mixture import sample_from_discretized_mix_logistic
from .wavenet import _expand_global_features, WaveNet

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow (parallel clarinet)
    """

    def __init__(self,
                cin_channels=-1, gin_channels=-1,
                n_speakers=None,
                upsample_conditional_features=False,
                upsample_scales=None,
                freq_axis_kernel_size=3,
                use_speaker_embedding=True,
                ):
        super(IAF, self).__init__()
        self.cin_channels = cin_channels

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                    n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # NOTE: hardcoding these hparams for now...
        l = [
                (10, 1),
                (10, 1),
                (10, 1),
                (10, 1),
                (10, 1),
                (10, 1),
            ]

        self.wavenet_stack = nn.ModuleList()
        for layers, stacks in l:
            self.wavenet_stack.append(
                    WaveNet(out_channels=2, layers=layers, stacks=stacks,
                        residual_channels=64,
                        gate_channels=64,
                        skip_out_channels=128,
                        kernel_size=3, dropout=1 - 0.95,
                        cin_channels=cin_channels,
                        gin_channels=gin_channels,
                        n_speakers=n_speakers,
                        weight_normalization=True,
                        upsample_conditional_features=upsample_conditional_features,
                        upsample_scales=upsample_scales,
                        freq_axis_kernel_size=freq_axis_kernel_size,
                        scalar_input=True,
                        use_speaker_embedding=use_speaker_embedding,
                        clarinet=True
                        ))

    def forward(self, z=None, c=None, g=None, softmax=False):
        # NOTE: softmax param is just here for training `model` compatibility.
        B, _, T = z.size()
        device = z.device

        self.loc = torch.zeros_like(z)
        self.log_scale = torch.zeros_like(z)
        x = z

        for w in self.wavenet_stack:
            o = w(x, c=c, g=g, softmax=False)
            s = o[:, 0, :].unsqueeze(1)
            l = o[:, 1, :].unsqueeze(1)

            s = torch.clamp(s, -7.0, 7.0)
            s_exp = torch.exp(s)
            x = x * s_exp + l
            self.loc = self.loc * s_exp + l
            self.log_scale = self.log_scale + s

        return z * torch.exp(self.log_scale) + self.loc

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

