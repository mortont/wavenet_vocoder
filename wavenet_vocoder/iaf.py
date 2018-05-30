# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

import torch
from torch import import nn
from torch.nn import functional as F

from .wavenet import WaveNet

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow (parallel wavenet)
    """

    def __init__(self, n_wavenets=4,
            out_channels=2, layers=20, stacks=2,
            residual_channels=512,
            gate_channels=512,
            skip_out_channels=512,
            kernel_size=3, dropout=1 - 0.95,
            cin_channels=-1, gin_channels=-1, n_speakers=None,
            weight_normalization=True,
            upsample_conditional_features=False,
            upsample_scales=None,
            freq_axis_kernel_size=3,
            scalar_input=False,
            use_speaker_embedding=True,
            ):
        # TODO: could pass lists of hparams for finer control of each
        # wavenet in the stack
        super(IAF, self).__init__()
        self.wavenet_stack = nn.ModuleList()
        for layer in range(n_wavenets):
            self.wavenet_stack.append(
                    WaveNet(out_channels=out_channels, layers=layers, stacks=stacks,
                        residual_channels=residual_channels,
                        gate_channels=gate_channels,
                        skip_out_channels=skip_out_channels,
                        kernel_size=kernel_size, dropout=dropout,
                        weight_normalization=weight_normalization,
                        cin_channels=cin_channels, gin_channels=gin_channels,
                        n_speakers=n_speakers,
                        upsample_conditional_features=upsample_conditional_features,
                        upsample_scales=upsample_scales,
                        freq_axis_kernel_size=freq_axis_kernel_size,
                        scalar_input=scalar_input,
                        use_speaker_embedding=use_speaker_embedding))

    def forward(self, x, c=None, g=None, softmax=False):
        # NOTE: softmax param is just here for training `model` compatibility.
        B, _, T = x.size()

        loc = torch.zeros(B, T)
        scale = torch.ones(B, T)

        z_dist = torch.distributions.normal.Normal(
                loc=loc,
                scale=scale)
        z = z_dist.sample((1,))

        for w in self.wavenet_stack:
            l, s = w(z, c=c, g=g, softmax=False)
            z = z * s + l
            loc = loc * s + l
            scale = scale * s

        z_tot = torch.distributions.normal.Normal(
                loc=loc,
                scale=scale)

        return z, z_tot
