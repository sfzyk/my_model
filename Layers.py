import torch
import torch.nn as  nn
from torch.nn import init

from SubLayers import WAN, FinalSoftmax, AspectAttention

__author__ = "Li Xi"


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, n_block,max_len, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.state = nn.Parameter(torch.FloatTensor(n_block, d_model))
        init.constant_(self.state, 0)

        self.slf_attn = WAN(n_head, d_model, d_k, d_v, dropout=dropout)
        # self.rnn = REN(d_model, n_block)
        self.sente_attn = AspectAttention(d_model, max_len,dropout=dropout)


    # enc_input consists word embedding and position embedding
    # enc_aspect consists aspect embedding
    def forward(self, enc_input, aspect_embedding, slf_attn_mask=None):
        out1, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        #out1 = self.rnn(enc_input, enc_output, self.state)
        out1 = self.sente_attn(out1, aspect_embedding)
        return out1


class DecoderLayer(nn.Module):
    def __init__(self, n_block, num_class, dropout):
        super(DecoderLayer, self).__init__()

        self.n_block = n_block
        self.num_class = num_class

        self.softmax = FinalSoftmax(self.n_block, self.num_class, dropout=dropout)

    def forward(self, inputs):
        outputs = self.softmax(inputs)
        return outputs
