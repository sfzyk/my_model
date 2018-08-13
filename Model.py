import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer

__author__ = "Li Xi"


class MyModel(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, n_block,num_class, dropout=0.1):
        super(MyModel, self).__init__()

        self.encoder = EncoderLayer(n_head, d_model, d_k, d_v, n_block, dropout=dropout)
        self.decoder = DecoderLayer(n_block,num_class)

    def forward(self, enc_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output)

        return dec_output
