"""
attention.py -- multi-head causal self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    """Every token looks at every earlier token, through several lenses at once.

    One head learns one way of relating tokens. Two independently initialised
    heads produce completely different attention patterns on the same sentence
    (proven in ch06), so a single head is a compromise rather than a
    simplification. n_heads of them run in parallel, each free to specialise.

    Input:  (B, seq_length, d_model)
    Output: (B, seq_length, d_model)
    """

    def __init__(self, config):
        super().__init__()

        self.d_k = config.d_k
        self.n_heads = config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.W_o = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )

        self.register_buffer("mask", mask, persistent=False)