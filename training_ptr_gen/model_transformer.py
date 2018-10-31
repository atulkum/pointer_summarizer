from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util import config
from numpy import random
import numpy as np

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

d_model = 512
N = 6
H = 8
d_k = d_v = 64 # = d_model / H
sqrt_d_k = np.sqrt(d_k)

d_ff = 2048

def get_pos_embedding(max_len):
    pos_emb = np.zeros(max_len, d_model)
    pos = np.arange(0, max_len)

    denom = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pos_emb[:, 0::2] = np.sin(pos*denom)
    pos_emb[:, 1::2] = np.cos(pos*denom)

    return pos_emb

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        layers = [EncoderLayer() for _ in range(N)]
        features = nn.Sequential(*layers)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input) * np.sqrt(d_model)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.multi_head_att = MultiHeadAttention()
        self.ln_mh = nn.LayerNorm([d_model])

        self.affine1 = nn.Linear(d_model, d_ff, bias=True)
        self.affine2 = nn.Linear(d_ff, d_model, bias=True)
        self.ln_aff = nn.LayerNorm([d_model])

    def forward(self, x):
        x_att = self.multi_head_att(x)
        x_1 = self.ln_mh(x + x_att)

        x_aff = F.relu(self.affine1(x_1))
        x_aff = self.affine2(x_aff)
        x_2 = self.ln_aff(x_1 + x_aff)

        return x_2

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v):
        q_proj = self.W_Q(q) #B x d_model
        k_proj = self.W_K(k)
        v_proj = self.W_V(v)

        q_proj_split = torch.split(q_proj, split_size_or_sections=d_k, dim=1)
        k_proj_split = torch.split(k_proj, split_size_or_sections=d_k, dim=1)
        v_proj_split = torch.split(v_proj, split_size_or_sections=d_v, dim=1)

        heads = []

        for i, q_i in enumerate(q_proj_split):
            k_i = k_proj_split[i] #B x d_k
            v_i = v_proj_split[i]
            qk_i = torch.bmm(q_i, k_i)/sqrt_d_k ##B x d_k
            att_i = F.softmax(qk_i)
            att_i = torch.bmm(att_i, v_i)

            heads.append(att_i)

        heads_cat = torch.cat(heads, dim=1)
        multi_head_att = self.W_O(heads_cat)

        return multi_head_att

