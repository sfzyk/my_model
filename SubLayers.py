import torch
import torch.nn as nn
from torch.nn import init, Linear, Tanh, Softmax
from Modules import ScaledDotProductAttention, LayerNormalization, RNN_cell

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

        self.cell = RNN_cell(self.n_block, self.d_model)

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

                current_state = self.cell(e_k, m_k, current_state)

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
                blocks.append(torch.sum(out_blocks[j]).view(1,1))


            out = torch.cat(blocks, 0)
            out = out.view(1, self.n_block)
            print(out.size())
            outputs.append(out)

        outputs = torch.cat(outputs, 1)
        outputs = outputs.view(bm_size, self.n_block)

        return outputs
