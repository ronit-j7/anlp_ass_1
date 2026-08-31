"""
attention.py -- attention primitives, built from scratch.

Contents
    scaled_dot_product_attention : the softmax(Q Kᵀ / sqrt(d_k)) V core
    MultiHeadAttention   (MHA)   : H query heads, H key/value heads
    GroupedQueryAttention (GQA)  : H query heads, G key/value heads (G | H, G <= H)

No nn.MultiheadAttention and no F.scaled_dot_product_attention are used.

Both attention blocks share one forward signature

    forward(x_q, x_kv, mask) -> (B, Tq, d_model)

so they are drop-in interchangeable inside an encoder/decoder layer:
    self-attention  -> pass x_kv = x_q
    cross-attention -> pass x_kv = encoder memory

Masking convention: `mask` is either
    * a bool tensor broadcastable to (B, H, Tq, Tk), True = "may attend", or
    * a float tensor broadcastable to the same shape, added to the scores
      (0 = keep, large negative = block).
These modules stay mask-agnostic; build causal / padding masks in utils.

RoPE: if a `rope` module is passed at construction it is applied to Q and K
inside the block. Only self-attention blocks get a rope; cross-attention
blocks are built with rope=None.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,                       # (B, H, Tq, Dh)
    k: torch.Tensor,                       # (B, H, Tk, Dh)
    v: torch.Tensor,                       # (B, H, Tk, Dh)
    mask: Optional[torch.Tensor] = None,   # bool or float, broadcastable to (B, H, Tq, Tk)
    dropout: Optional[nn.Dropout] = None,  # applied to the post-softmax weights
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(Dh)) V.

    Returns (context, weights):
        context : (B, H, Tq, Dh)
        weights : (B, H, Tq, Tk)   post-softmax, post-dropout
    """
    d_head = q.size(-1)

    # Similarity scores scaled by 1/sqrt(Dh): the dot product of two random
    # Dh-vectors has variance ~ Dh, so without this the softmax saturates as
    # Dh grows and gradients through it vanish.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)   # (B, H, Tq, Tk)

    if mask is not None:
        if mask.dtype == torch.bool:
            # finfo.min, not -inf: a fully-blocked query row would softmax to
            # NaN under -inf. finfo.min gives a ~uniform row instead, which is
            # harmless because such rows are padding and dropped by the loss.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask                            # additive mask

    weights = torch.softmax(scores, dim=-1)                  # (B, H, Tq, Tk)
    if dropout is not None:
        weights = dropout(weights)

    context = weights @ v                                    # (B, H, Tq, Dh)
    return context, weights


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention.

    `n_heads` independent Q/K/V projections of width d_head = d_model // n_heads
    run in parallel, are concatenated, then mixed by an output projection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope = rope

        # Each maps d_model -> d_model (= n_heads * d_head).
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, n_heads, T, d_head) -> (B, T, d_model)
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(
        self,
        x_q: torch.Tensor,                    # (B, Tq, d_model)
        x_kv: torch.Tensor,                   # (B, Tk, d_model)   (== x_q for self-attn)
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, Tq, Tk)
        return_weights: bool = False,
    ):
        q = self._split_heads(self.w_q(x_q))   # (B, H, Tq, Dh)
        k = self._split_heads(self.w_k(x_kv))  # (B, H, Tk, Dh)
        v = self._split_heads(self.w_v(x_kv))  # (B, H, Tk, Dh)

        # Rotary embeddings rotate Q and K before the dot product. Present on
        # self-attention only (cross-attention blocks pass rope=None).
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )
        out = self.w_o(self._merge_heads(context))   # (B, Tq, d_model)
        return (out, weights) if return_weights else out


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention.

    `n_heads` query heads but only `n_kv_heads` key/value heads, with
    n_kv_heads dividing n_heads. Each K/V head is shared by a group of
    n_heads // n_kv_heads query heads, shrinking the K/V projections and
    (with a KV cache at decode time) the KV memory / bandwidth, for a small
    quality cost.

        n_kv_heads == n_heads  ->  equivalent to MHA
        n_kv_heads == 1        ->  multi-query attention (MQA)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads      # query heads per K/V head
        self.d_head = d_model // n_heads
        self.rope = rope

        # Q keeps full width; K and V are narrower: n_kv_heads * d_head.
        self.w_q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.w_o = nn.Linear(n_heads * self.d_head, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,                    # (B, Tq, d_model)
        x_kv: torch.Tensor,                   # (B, Tk, d_model)
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, Tq, Tk)
        return_weights: bool = False,
    ):
        b, tq, _ = x_q.shape
        tk = x_kv.size(1)

        # (B, T, n*Dh) -> (B, n, T, Dh)
        q = self.w_q(x_q).view(b, tq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x_kv).view(b, tk, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x_kv).view(b, tk, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # Expand each K/V head across its group so head dims line up with Q.
        # repeat_interleave (not repeat) keeps a head next to its own copies:
        # kv head i -> query heads [i*g .. i*g+g-1].
        k = k.repeat_interleave(self.n_groups, dim=1)   # (B, n_heads, Tk, Dh)
        v = v.repeat_interleave(self.n_groups, dim=1)

        context, weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.attn_dropout
        )
        context = (
            context.transpose(1, 2).contiguous().view(b, tq, self.n_heads * self.d_head)
        )
        out = self.w_o(context)                         # (B, Tq, d_model)
        return (out, weights) if return_weights else out
