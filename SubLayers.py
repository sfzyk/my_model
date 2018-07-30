import torch
import torch.nn as nn
from torch.nn import init

from Modules import ScaledDotProductAttention

__author__ = "Li Xi"


class WAN(nn.Module):
    '''WAN module'''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(WAN, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v =d_v

        self.w_query = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_key = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_value = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)

        self.dropout = nn.Dropout(dropout)


        init.xavier_normal_(self.w_query)
        init.xavier_normal_(self.w_key)
        init.xavier_normal_(self.w_value)


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
        q_s = torch.bmm(q_s, self.w_query).view(-1, len_q, d_k)  # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_key).view(-1, len_k, d_k)  # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_value).view(-1, len_v, d_v)  # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        # outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        # With attn_mask is none, it can not use repeat here in the latest torch
        outputs, attns = self.attention(q_s, k_s, v_s)

        print(outputs.size())
        print(attns.size())

        return outputs, attns