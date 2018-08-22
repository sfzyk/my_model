import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, Linear, Tanh, Softmax, Sigmoid, PReLU

from Modules import ScaledDotProductAttention, LayerNormalization

__author__ = "Li Xi"



class WAN(nn.Module):
    '''WAN module'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(WAN, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_query = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.W_key = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.W_value = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal_(self.W_query)
        init.xavier_normal_(self.W_key)
        init.xavier_normal_(self.W_value)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)  # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.matmul(q_s, self.W_query).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.matmul(k_s, self.W_key).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.matmul(v_s, self.W_value).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        # outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        # With attn_mask is none, it can not use repeat here in the latest torch
        outputs, attns = self.attention(q_s, k_s, v_s)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(outputs + residual), attns


class REN(nn.Module):
    '''REN module'''

    def __init__(self, d_model, n_block):
        super(REN, self).__init__()

        self.d_model = d_model
        self.n_block = n_block

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

    def rnn_cell(self,e,m,state):

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

    def forward(self, e, m, state):
        '''e is the word-pos-embedding; m is the output of WAN; '''

        bm_size, n_len, embed_size = e.size()

        batch_e = torch.chunk(e, bm_size, 0)
        batch_m = torch.chunk(m, bm_size, 0)
        outputs = []

        for i in range(len(batch_e)):

            e_s = torch.chunk(batch_e[i].view(n_len, embed_size), n_len, 0)
            m_s = torch.chunk(batch_m[i].view(n_len, embed_size), n_len, 0)

            current_state = state

            for k in range(len(e_s)):
                e_k = e_s[k]
                m_k = m_s[k]

                current_state = self.rnn_cell(e_k, m_k, current_state)



            outputs.append(current_state)

        outputs = torch.cat(outputs, 1)
        outputs = outputs.view(bm_size, self.n_block, self.d_model)

        return outputs


class AspectAttention(nn.Module):
    '''Aspect attention module'''

    def __init__(self, d_model, n_block):
        super(AspectAttention, self).__init__()
        self.d_model = d_model
        self.n_block = n_block

        self.tanh = Tanh()
        self.softmax = Softmax(dim=1)

        self.W_a = nn.Parameter(torch.FloatTensor(self.d_model, self.d_model))
        self.b_a = nn.Parameter(torch.FloatTensor(self.n_block, self.d_model))

        init.xavier_normal_(self.W_a)
        init.constant_(self.b_a, 0)

    def forward(self, h):
        bm_size, num_block, embed_size = h.size()

        batch_h = torch.chunk(h, bm_size, 0)

        outputs = []

        for i in range(len(batch_h)):
            out = torch.matmul(batch_h[i], self.W_a) + self.b_a
            out = self.tanh(out)
            out = self.softmax(out)
            out = out * h[i]
            out = out.view(self.n_block, self.d_model)

            out_blocks = torch.chunk(out, self.n_block, 0)
            blocks = []
            for j in range(len(out_blocks)):
                blocks.append(torch.sum(out_blocks[j]).view(1, 1))

            out = torch.cat(blocks, 0)
            out = out.view(1, self.n_block)
            outputs.append(out)

        outputs = torch.cat(outputs, 1)
        outputs = outputs.view(bm_size, self.n_block)

        return outputs


class FinalSoftmax(nn.Module):

    def __init__(self, n_block, num_class):
        super(FinalSoftmax, self).__init__()

        self.n_block = n_block
        self.num_class = num_class

        self.linear = Linear(self.n_block, self.num_class)
        self.softmax = Softmax(dim=1)

    def forward(self, sente):
        out = self.linear(sente)
        out = self.softmax(out)

        return out
