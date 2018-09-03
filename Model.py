import torch
import torch.nn as nn

from Layers import EncoderLayer, DecoderLayer

__author__ = "Li Xi"


class MyModel(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, n_block, num_class, max_len, dropout=0.1):
        super(MyModel, self).__init__()

        self.encoder = EncoderLayer(n_head, d_model, d_k, d_v, n_block, max_len, dropout=dropout)
        self.decoder = DecoderLayer(max_len, num_class,dropout=dropout)

    def forward(self, enc_input, aspect_embedding=None):
        #bm_size, n_len, vec = enc_input.size()
        #aspect_embedding = nn.Parameter(torch.FloatTensor(bm_size,1, vec))
        enc_output = self.encoder(enc_input, aspect_embedding)
        dec_output = self.decoder(enc_output)

        return dec_output
