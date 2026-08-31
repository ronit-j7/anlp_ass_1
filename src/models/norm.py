"""
norm.py -- normalization layers, built from scratch.

    LayerNorm : subtract the mean, divide by the std over the feature dim,
                then a learned per-feature scale (gamma) and shift (beta).
    RMSNorm   : divide by the root-mean-square over the feature dim, then a
                learned per-feature scale (gamma).  No mean subtraction, no shift.

Both normalize over the last axis (the d_model features), independently for
every (batch, position) pair. "Pre-norm" vs "post-norm" is about *where* the
norm is called inside a block -- that decision lives in transformer.py; here we
only define the op.  C1/C2/C3/C5 use LayerNorm, C4 swaps in RMSNorm.

Statistics are computed in fp32 and cast back, so behaviour is stable under
autocast / bf16 (matches how torch.nn.LayerNorm handles mixed precision).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta,  over the last dim.

    Mirrors torch.nn.LayerNorm: population variance (unbiased=False), affine
    parameters enabled.  Reimplemented because the assignment asks for the
    norm modules from scratch.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma
        self.bias = nn.Parameter(torch.zeros(dim))    # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)          # 1/N, not 1/(N-1)
        normed = (x - mean) * torch.rsqrt(var + self.eps)
        return normed.to(orig_dtype) * self.weight + self.bias


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma,  over the last dim.

    Drops LayerNorm's mean-centering and its beta shift.  The argument
    (Zhang & Sennrich, 2019) is that re-scaling, not re-centering, is what
    stabilizes training, so RMSNorm keeps only that -- fewer ops, one fewer
    parameter vector.  This is C4's single change vs the C1 base.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms_inv).to(orig_dtype) * self.weight
