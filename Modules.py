import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, Sigmoid, PReLU

__author__ = "Li Xi"

class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])

class BottleLayerNormalization(BatchBottle, LayerNormalization):
    ''' Perform the reshape routine before and after a layer normalization'''
    pass

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

class RNN_cell(nn.Module):
    '''RNN cell'''

    def __init__(self, n_block, d_model):
        super(RNN_cell, self).__init__()

        self.n_block = n_block
        self.d_model = d_model

        self.keys = nn.Parameter(torch.FloatTensor(self.n_block, self.d_model))
        self.V_m = nn.Parameter(torch.FloatTensor(1, self.d_model))

        self.U = nn.Parameter(torch.FloatTensor(self.d_model, self.d_model))
        self.U_bias = nn.Parameter(torch.FloatTensor(self.d_model))
        self.V = nn.Parameter(torch.FloatTensor(self.d_model, self.d_model))
        self.W = nn.Parameter(torch.FloatTensor(self.d_model, self.d_model))

        self.sigmoid = Sigmoid()
        self.prelu = PReLU()

        # The keys can be initialed with embedding
        # Here we simpliy initial with xavier_normal_
        init.xavier_normal_(self.keys)

        init.xavier_normal_(self.V_m)
        init.xavier_normal_(self.U)
        init.constant_(self.U_bias, 0)
        init.xavier_normal_(self.V)
        init.xavier_normal_(self.W)

    def get_gate(self, state_j, key_j, e, m):
        # To reservethe  orgin form of the tensor, it is necessary to clone it before transpose
        e_t = e.clone()
        m_t = m.clone()
        e_t.t_()
        m_t.t_()

        # 1 x 1  be cautious about the order of the parameters
        a = torch.matmul(key_j, e_t)
        b = torch.matmul(state_j, e_t)
        c = torch.matmul(self.V_m, m_t)

        return self.sigmoid(a + b + c)

    def get_candidate(self, state_j, key_j, e, U, V, W, U_bias):
        # 1 x embed_size
        key_V = torch.matmul(key_j, V)
        state_U = torch.matmul(state_j, U) + U_bias
        input_W = torch.matmul(e, W)

        return self.prelu(state_U + key_V + input_W)

    def forward(self, e, m, state):
        n_block, embed_size = state.size()
        state = torch.chunk(state, n_block, 0)

        next_states = []
        for j in range(n_block):
            key_j = torch.unsqueeze(self.keys[j], 0)
            gate_j = self.get_gate(state[j], key_j, e, m)
            candidate_j = self.get_candidate(state[j], key_j, e, self.U, self.V, self.W, self.U_bias)
            state_j_next = state[j] + gate_j * candidate_j
            next_states.append(state_j_next)
        state_next = torch.cat(next_states, 1)
        state_next = state_next.view(self.n_block, self.d_model)

        return state_next