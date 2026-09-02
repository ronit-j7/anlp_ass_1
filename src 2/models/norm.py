"""Normalization layers implemented from scratch."""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, over the last dim."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma. No mean subtraction, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def make_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"unknown norm: {kind}")
