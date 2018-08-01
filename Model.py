import torch
import torch.nn as nn

from Layers import EncoderLayer

__author__ = "Li Xi"


class MyModel(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v,n_block, dropout=0.1):
        super(MyModel, self).__init__()

        self.encoder = EncoderLayer(n_head, d_model, d_k, d_v, n_block, dropout=dropout)

    def forward(self, enc_input):


        out = self.encoder(enc_input)

        return out

