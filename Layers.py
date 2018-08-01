import torch

import torch.nn as  nn
from torch.nn import init

from SubLayers import WAN, REN

__author__ = "Li Xi"

class EncoderLayer(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, n_block, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.state = nn.Parameter(torch.FloatTensor(n_block, d_model))
        init.constant_(self.state, 0)

        self.slf_attn = WAN(n_head, d_model, d_k, d_v, dropout=dropout)
        self.rnn = REN(d_model,n_block)
    def forward(self, enc_input,slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)

        out1, out2 = self.rnn(enc_input, enc_output, self.state)

        return out1, out2

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self, *input):

        return None

