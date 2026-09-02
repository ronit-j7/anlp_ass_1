"""Attention built from basic tensor ops: scaled dot-product, MHA and GQA."""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RotaryPositionalEmbedding


def scaled_dot_product_attention(q, k, v, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Dropout] = None):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    q: (B, H, Lq, hd), k/v: (B, H, Lk, hd)
    mask: additive float mask broadcastable to (B, H, Lq, Lk); 0 keeps, -inf drops.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention. With n_kv_heads < n_heads this is Grouped-Query Attention:
    the n_heads queries are split into n_kv_heads groups that share one K/V head."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert d_model % n_heads == 0 and n_heads % self.n_kv_heads == 0
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.rope = rope

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x, n_heads):
        B, L, _ = x.shape
        return x.view(B, L, n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, hd)

    def forward(self, query, key_value, mask: Optional[torch.Tensor] = None):
        B, Lq, _ = query.shape
        q = self._split(self.w_q(query), self.n_heads)
        k = self._split(self.w_k(key_value), self.n_kv_heads)
        v = self._split(self.w_v(key_value), self.n_kv_heads)

        if self.rope is not None:  # self-attention only (Lq == Lk)
            q, k = self.rope(q, k)

        if self.n_rep > 1:  # share each K/V head across its query group
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.head_dim)
        return self.w_o(out)


class GroupedQueryAttention(MultiHeadAttention):
    """Thin alias: MHA with fewer K/V heads than query heads."""

    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, rope=None):
        assert n_kv_heads < n_heads, "GQA requires n_kv_heads < n_heads"
        super().__init__(d_model, n_heads, n_kv_heads, dropout, rope)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
