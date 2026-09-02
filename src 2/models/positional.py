"""Positional information: sinusoidal absolute encodings and RoPE."""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed absolute encoding added to token embeddings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return self.dropout(x + self.pe[: x.size(1)].unsqueeze(0))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: rotates (q, k) pairs of dims by an angle proportional to position.

    Applied inside self-attention, per head, instead of adding an absolute encoding.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)            # (max_len, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, hd) -> split even/odd dims and rotate each 2D pair
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        T = q.size(-2)
        cos = self.cos[:T].to(q.dtype).view(1, 1, T, -1)
        sin = self.sin[:T].to(q.dtype).view(1, 1, T, -1)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
