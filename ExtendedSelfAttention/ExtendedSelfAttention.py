
import torch
import torch.nn as nn
import torch.optim as optim
import transformers

import os
from transformers import AutoTokenizer

load_dotenv()

class ExtendedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(ExtendedSelfAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(d_model, d_model * n_heads)
        self.k_linear = nn.Linear(d_model, d_model * n_heads)
        self.v_linear = nn.Linear(d_model, d_model * n_heads)

        self.attention_map = nn.Linear(d_model * n_heads, d_model)

    def forward(self, query, key, value, mask=None):
        # Calculate the attention scores.
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(query.size(0), query.size(1), self.n_heads, self.d_model // self.n_heads)
        k = k.view(key.size(0), key.size(1), self.n_heads, self.d_model // self.n_heads)
        v = v.view(value.size(0), value.size(1), self.n_heads, self.d_model // self.n_heads)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_model // self.n_heads))

        # Apply the mask.
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))

        # Calculate the attention weights.
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Calculate the weighted context vector.
        context = torch.matmul(attention_weights, v)

        # Calculate the output.
        output = self.attention_map(context.view(context.size(0), context.size(1), self.d_model))

        return output


