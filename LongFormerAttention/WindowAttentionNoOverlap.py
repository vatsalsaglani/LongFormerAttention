from longformer.sliding_chunks import pad_to_window_size, sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class WindowAttentionNoOverlap(nn.Module):

    def __init__(self, window_size, padding_id = 0, dropout = 0.1):

        super().__init__()

        self.window_size = window_size
        self.padding_id = padding_id

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):

        attention = sliding_chunks_no_overlap_matmul_qk(query, key, self.window_size, self.padding_id)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = self.dropout(torch.softmax(attention, dim = -1))

        output = sliding_chunks_no_overlap_matmul_pv(attention, value, self.window_size)

        return output, attention

