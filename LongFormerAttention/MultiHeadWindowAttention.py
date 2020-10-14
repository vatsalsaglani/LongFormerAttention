import torch 
import torch.nn as nn
import torch.nn.functional as F
from .WindowAttentionNoOverlap import WindowAttentionNoOverlap
import numpy as np
import os

class MultiHeadWindowAttention(nn.Module):

    def __init__(self, heads, window_size, dim_model, dim_key, dim_value, dropout = 0.1, padding_id = 0):

        super().__init__()

        self.heads = heads
        self.dim_key = dim_key
        self.dim_value = dim_value 
        self.window_size = window_size

        self.toquery = nn.Linear(dim_model, heads * dim_key, bias = False)
        self.tokey = nn.Linear(dim_model, heads * dim_key, bias = False)
        self.tovalue = nn.Linear(dim_model, heads * dim_value, bias = False)

        self.union = nn.Linear(heads * dim_value, dim_model, bias = False)

        self.attention = WindowAttentionNoOverlap(self.window_size, self.padding_id, dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(dim_model, eps = 1e-6)

    def forward(self, query, key, value, mask = None):

        dim_key, dim_value, heads = self.dim_key, self.dim_value, self.heads

        batch_size, length_query, length_key, length_value = query.size(0), query.size(1), key.size(0), key.size(1), value.size(1)

        residual = query

        query = self.layer_norm(query)

        query = self.toquery(query).view(batch_size, length_query, heads, dim_key)
        key = self.tokey(key).view(batch_size, length_key, heads, dim_key)
        value = self.tovalue(value).view(batch_size, length_value, heads, dim_value)

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        query, attention = self.attention(query, key, value, mask = mask)

        query = query.transpose(1, 2).contiguous().view(batch_size, length_query, -1)

        query = self.dropout(self.union(query))

        query += residual

        return query, attention
