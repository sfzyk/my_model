import torch.nn as  nn

from SubLayers import WAN

__author__ = "Li Xi"

class EncoderLayer(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = WAN(n_head, d_model, d_k, d_v, dropout=0.1)

    def forward(self, enc_input,slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)


        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self, *input):

        return None

