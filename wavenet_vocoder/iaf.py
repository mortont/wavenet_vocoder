# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .modules import Embedding
from .modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from .mixture import sample_from_discretized_mix_logistic
from .wavenet import _expand_global_features

class IAF(nn.Module):
    """
    Inverse Autoregressive Flow (parallel wavenet)
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
        l = [(10, 1), (10, 1), (10, 1), (30, 3)]
        self.wavenet_stack = nn.ModuleList()
        for layers, stacks in l:
            self.wavenet_stack.append(
                    PWavenet(layers=layers, stacks=stacks,
                        cin_channels=cin_channels,
                        gin_channels=gin_channels,
                        n_speakers=n_speakers,
                        upsample_conditional_features=upsample_conditional_features,
                        upsample_scales=upsample_scales,
                        freq_axis_kernel_size=freq_axis_kernel_size,
                        use_speaker_embedding=use_speaker_embedding
                        ))

    def forward(self, x=None, c=None, g=None, softmax=False, lengths=None):
        # NOTE: softmax param is just here for training `model` compatibility.

        if x is None and lengths is not None:
            B = lengths.size()
            T = max(lengths)
        elif x is not None:
            B, _, T = x.size()
        else:
            raise RuntimeError("Must have either x or lengths")

        # TODO: make these inherit the model's device
        # https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/3
        loc = torch.nn.Parameter(torch.zeros(B, T)).cuda()
        scale = torch.nn.Parameter(torch.ones(B, T)).cuda()

        u_dist = torch.distributions.normal.Normal(
                loc=loc,
                scale=scale)
        # TODO: take multiple samples and average?
        z = u_dist.sample((1,))
        # Need to change batch size here if multiple samples
        z = z.squeeze(0)
        z = z.unsqueeze(1)

        for w in self.wavenet_stack:
            o = w(z, c=c, g=g, softmax=False)
            s = o[:, 0, :].unsqueeze(1)
            l = o[:, 1, :].unsqueeze(1)
            z = z * s + l
            loc = loc * s + l
            scale = scale * s

        self.z_dist = torch.distributions.normal.Normal(
                loc=loc,
                scale=scale)

        return z

    def out_dist(self):
        return self.z_dist

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

class PWavenet(nn.Module):
    """
    Wavenet used for stacking in the Inverse Autoregressive Flow. Separate
    from Autoregressive Wavenet since this doesn't use skip connections.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vector. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
        use_speaker_embedding (Bool): Use speaker embedding or Not. Set to False
          if you want to disable embedding layer and use external features
          directly.
    """
    # TODO: Maybe combine the similar parts of this and Wavenet into a parent type
    # for better code reuse.

    def __init__(self, out_channels=2, layers=10, stacks=1,
                 residual_channels=64,
                 gate_channels=64,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=80, gin_channels=-1, n_speakers=None,
                 weight_normalization=True,
                 upsample_conditional_features=True,
                 upsample_scales=[4, 4, 4 ,4],
                 freq_axis_kernel_size=3,
                 scalar_input=True,
                 use_speaker_embedding=True,
                 ):
        super(PWavenet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = Conv1d1x1(1, residual_channels)
        else:
            self.first_conv = Conv1d1x1(out_channels, residual_channels)

        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualConv1dGLU(
                residual_channels, gate_channels,
                kernel_size=kernel_size,
                skip_out_channels=None,
                bias=True,  # magenda uses bias, but musyoku doesn't
                dilation=dilation, dropout=dropout,
                cin_channels=cin_channels,
                gin_channels=gin_channels,
                weight_normalization=weight_normalization)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            Conv1d1x1(residual_channels, residual_channels,
                weight_normalization=weight_normalization),
            nn.ReLU(inplace=True),
            Conv1d1x1(residual_channels, out_channels,
                weight_normalization=weight_normalization)
        ])

        if gin_channels > 0 and use_speaker_embedding:
            assert n_speakers is not None
            self.embed_speakers = Embedding(
                n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # Upsample conv net
        if upsample_conditional_features:
            self.upsample_conv = nn.ModuleList()
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, x, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            g (Tensor): Global conditioning features,
              shape (B x gin_channels x 1) or speaker Ids of shape (B x 1).
              Note that ``self.use_speaker_embedding`` must be False when you
              want to disable embedding layer and use external features
              directly (e.g., one-hot vector).
              Also type of input tensor must be FloatTensor, not LongTensor
              in case of ``self.use_speaker_embedding`` equals False.
            softmax (bool): Whether applies softmax or not.

        Returns:
            Tensor: output, shape B x out_channels x T
        """
        try:
            B, _, T = x.size()
        except ValueError:
            # Handle eval case
            x = x.unsqueeze(0)
            x = x.transpose(-1, -2)
            B, _, T = x.size()

        if g is not None:
            if self.embed_speakers is not None:
                # (B x 1) -> (B x 1 x gin_channels)
                g = self.embed_speakers(g.view(B, -1))
                # (B x gin_channels x 1)
                g = g.transpose(1, 2)
                assert g.dim() == 3
        # Expand global conditioning features to all time steps
        g_bct = _expand_global_features(B, T, g, bct=True)

        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
            assert c.size(-1) == x.size(-1)

        # Feed data to network
        x = self.first_conv(x)
        for f in self.conv_layers:
            x, _ = f(x, c, g_bct)

        for f in self.last_conv_layers:
            x = f(x)

        x = F.softmax(x, dim=1) if softmax else x

        return x
