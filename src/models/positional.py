"""
positional.py -- position information for the transformer.

Contents
    SinusoidalPositionalEncoding : fixed sin/cos table, ADDED to token
                                   embeddings once at the bottom of the stack
                                   (Vaswani et al. 2017).  Used by C1, C3, C4, C5.
    RotaryPositionalEmbedding    : RoPE -- rotates Q and K inside every
                                   self-attention block; nothing is added to the
                                   embeddings.  Used by C2.

Why two different call sites:
    * Sinusoidal is an absolute signal, so it only needs to be injected once.
      forward(x): x is (B, T, d_model).
    * RoPE makes the Q·K dot product depend only on the *relative* offset
      (m - n), so it must be applied to Q and K every layer, per head.
      forward(x): x is (B, H, T, d_head).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) absolute positional encoding.

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    Each dimension is a sinusoid whose wavelength grows geometrically from
    ~2pi to ~10000*2pi, so every position gets a unique, smoothly varying
    fingerprint and relative offsets are expressible as a linear mix of these
    dimensions.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for the sin/cos interleave"
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)        # (max_len, 1)
        # div_term[i] = 10000^(-2i / d_model), computed in log space for stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )                                                                   # (d_model/2,)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # persistent=False: recomputable constant, keep it out of state_dict.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)       # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        t = x.size(1)
        if t > self.pe.size(1):
            raise ValueError(f"sequence length {t} exceeds max_len {self.pe.size(1)}")
        x = x + self.pe[:, :t].to(x.dtype)
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding (RoPE).

    Splits each head vector into d_head/2 pairs and rotates pair i by angle
    pos * theta_i, with theta_i = base^(-2i / d_head). After rotation,
    q_m · k_n depends on m and n only through (m - n) -- position becomes
    relative, for free, with no extra parameters.

    Convention: the "rotate the two halves" form (LLaMA / GPT-NeoX), i.e.
    x is split as [x1 | x2] and rotate_half(x) = [-x2 | x1]. Pairs are
    (x1[i], x2[i]), not adjacent elements.
    """

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )                                                                  # (d_head/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute cos/sin for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                 # (seq_len, d_head/2)
        emb = torch.cat((freqs, freqs), dim=-1)               # (seq_len, d_head)
        # shape (1, 1, seq_len, d_head) so it broadcasts over (B, H, T, d_head)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate x by its position angles.

        x      : (B, H, T, d_head)  -- a Q or K tensor
        offset : position of x[..., 0, :] in the full sequence (0 unless a
                 KV cache is feeding one token at a time; we recompute the
                 whole prefix each decode step, so it stays 0 here).
        """
        t = x.size(-2)
        if offset + t > self._cached_len:
            self._build_cache(offset + t)
        cos = self.cos_cached[..., offset:offset + t, :].to(x.dtype)
        sin = self.sin_cached[..., offset:offset + t, :].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin
