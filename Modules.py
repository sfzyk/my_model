import numpy as np

import torch
import torch.nn as nn
from torch.nn import init


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout= 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn