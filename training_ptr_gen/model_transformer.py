#This is still a work in progress I will work on it once I get some free time.

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logging.basicConfig(level=logging.INFO)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head  #d_k == d_v
        self.h = num_head

        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask):
        nbatches = query.size(0)
        query = self.linear_query(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_key(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear_out(x)

class AffineLayer(nn.Module):
    def __init__(self, dropout, d_model, d_ff):
        super(AffineLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, num_head, dropout, d_model, d_ff):
        super(EncoderLayer, self).__init__()

        self.att_layer = MultiHeadedAttention(num_head, d_model, dropout)
        self.norm_att = nn.LayerNorm(d_model)
        self.dropout_att = nn.Dropout(dropout)

        self.affine_layer = AffineLayer(dropout, d_model, d_ff)
        self.norm_affine = nn.LayerNorm(d_model)
        self.dropout_affine = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_att = self.norm_att(x*mask)
        x_att = self.att_layer(x_att, x_att, x_att, mask)
        x = x + self.dropout_att(x_att)

        x_affine = self.norm_affine(x*mask)
        x_affine = self.affine_layer(x_affine)
        return x + self.dropout_affine(x_affine)

class Encoder(nn.Module):
    def __init__(self, N, num_head, dropout, d_model, d_ff):
        super(Encoder, self).__init__()
        self.position = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList()
        for _ in range(N):
            self.layers.append(EncoderLayer(num_head, dropout, d_model, d_ff))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, word_embed, mask):
        x = self.position(word_embed)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x*mask)
