"""Byte Latent Transformer: entropy-based dynamic patching + local byte encoder/decoder.

Patch boundaries are placed where the next-byte entropy of a small n-gram model exceeds a
threshold (BLT, Pagnoni et al. 2025), so patches are dynamic, not fixed width. The local
encoder turns each patch of bytes into one latent, the global transformer works on latents,
and the local decoder expands a latent back into bytes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import make_norm
from .positional import SinusoidalPositionalEncoding
from .transformer import (NEG_INF, Decoder, DecoderLayer, Encoder, EncoderLayer, ModelConfig,
                          causal_mask, init_weights, pad_mask)

V = 256          # byte values
SEP = 256        # sequence separator used while fitting the n-gram counts


class ByteEntropyModel:
    """Order-2 next-byte n-gram model with back-off, used only to place patch boundaries."""

    def __init__(self, alpha: float = 0.1, min_count: int = 8):
        self.alpha, self.min_count = alpha, min_count

    @staticmethod
    def _entropy(counts, alpha):
        p = (counts + alpha) / (counts.sum(-1, keepdims=True) + V * alpha)
        return -(p * np.log(p)).sum(-1)

    def fit(self, sequences):
        arr = np.concatenate([np.concatenate([[SEP, SEP], np.asarray(s, dtype=np.int64)])
                              for s in sequences])
        a, b, x = arr[:-2], arr[1:-1], arr[2:]
        ok0 = x != SEP
        c0 = np.bincount(x[ok0], minlength=V).astype(np.float64)
        ok1 = ok0 & (b != SEP)
        c1 = np.bincount(b[ok1] * V + x[ok1], minlength=V * V).reshape(V, V).astype(np.float64)
        ok2 = ok1 & (a != SEP)
        c2 = np.bincount((a[ok2] * V + b[ok2]) * V + x[ok2],
                         minlength=V * V * V).reshape(V * V, V).astype(np.float64)
        self.h0 = float(self._entropy(c0, self.alpha))
        self.h1, self.n1 = self._entropy(c1, self.alpha), c1.sum(-1)
        self.h2, self.n2 = self._entropy(c2, self.alpha), c2.sum(-1)
        return self

    def entropy_at(self, prev2, prev1):
        """Vectorised H(x_i | x_{i-2}, x_{i-1}) with back-off; a context >= 256 is 'unknown'."""
        prev1, prev2 = np.asarray(prev1), np.asarray(prev2)
        h = np.full(prev1.shape, self.h0)
        ok1 = prev1 < V
        i1 = np.where(ok1, prev1, 0)
        h = np.where(ok1 & (self.n1[i1] >= self.min_count), self.h1[i1], h)
        ok2 = ok1 & (prev2 < V)
        i2 = np.where(ok2, prev2, 0) * V + i1
        return np.where(ok2 & (self.n2[i2] >= self.min_count), self.h2[i2], h)

    def entropies(self, seq):
        s = np.asarray(seq, dtype=np.int64)
        prev1 = np.concatenate([[SEP], s[:-1]])
        prev2 = np.concatenate([[SEP, SEP], s[:-2]])
        return self.entropy_at(prev2, prev1)

    def segments(self, seq, theta):
        """Patch id per byte: a new patch starts wherever the next-byte entropy exceeds theta."""
        cut = self.entropies(seq) > theta
        cut[0] = True
        return np.cumsum(cut) - 1

    def calibrate(self, sequences, target_patch):
        """Threshold giving an average patch length of ~target_patch bytes."""
        h = np.concatenate([self.entropies(s) for s in sequences])
        return float(np.percentile(h, 100 * (1 - 1 / target_patch)))


# --------------------------------------------------------------------------- helpers
def patch_mask(seg, causal=False):
    """Additive mask allowing attention only between bytes of the same patch."""
    same = seg[:, :, None].eq(seg[:, None, :])
    if causal:
        T = seg.size(1)
        same = same & torch.ones(T, T, dtype=torch.bool, device=seg.device).tril()
    return torch.zeros(same.shape, device=seg.device).masked_fill(~same, NEG_INF).unsqueeze(1)


def pool_by_patch(x, seg, n_patches):
    """Mean-pool the bytes of each patch -> one vector per patch."""
    onehot = F.one_hot(seg, n_patches).to(x.dtype)                       # (B, T, M)
    return onehot.transpose(1, 2) @ x / onehot.sum(1).clamp(min=1).unsqueeze(-1)


class LocalEncoder(nn.Module):
    """Bytes -> one latent per patch; attention stays inside a patch."""

    def __init__(self, cfg: ModelConfig, vocab: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model, padding_idx=pad_id)
        self.pe = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_local_layers)])
        self.norm = make_norm(cfg.norm, cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.scale = cfg.d_model ** 0.5

    def forward(self, byte_ids, seg, n_patches):
        x = self.pe(self.emb(byte_ids) * self.scale)
        mask = patch_mask(seg)
        for layer in self.layers:
            x = layer(x, mask)
        return self.proj(pool_by_patch(self.norm(x), seg, n_patches))


class LocalDecoder(nn.Module):
    """Patch latents + byte history -> byte logits, autoregressive inside each patch."""

    def __init__(self, cfg: ModelConfig, vocab: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, cfg.d_model, padding_idx=pad_id)
        self.pe = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_local_layers)])
        self.norm = make_norm(cfg.norm, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.scale = cfg.d_model ** 0.5

    def forward(self, byte_ids_in, seg, latents):
        x = self.pe(self.emb(byte_ids_in) * self.scale)
        self_mask = patch_mask(seg, causal=True)
        # byte i cross-attends to the latents of its own patch and of all preceding patches
        idx = torch.arange(latents.size(1), device=seg.device)
        cross = torch.zeros(seg.size(0), seg.size(1), latents.size(1), device=seg.device)
        cross = cross.masked_fill(idx[None, None, :] > seg[:, :, None], NEG_INF).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, latents, self_mask, cross)
        return self.head(self.norm(x))


class BLTSeq2Seq(nn.Module):
    """Token-free encoder-decoder over raw bytes (configuration C5)."""

    ENCODE_KEYS = ("src", "src_seg")

    def __init__(self, cfg: ModelConfig, vocab, pad_id, bos_id, eos_id, patcher=None, theta=0.0):
        super().__init__()
        self.cfg, self.pad_id, self.bos_id, self.eos_id = cfg, pad_id, bos_id, eos_id
        self.src_local = LocalEncoder(cfg, vocab, pad_id)
        self.tgt_local = LocalEncoder(cfg, vocab, pad_id)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.local_decoder = LocalDecoder(cfg, vocab, pad_id)
        self.patch_pe = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.bos_patch = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        init_weights(self, cfg.d_model)
        self.local_decoder.head.weight = self.local_decoder.emb.weight   # keep tying after init
        nn.init.normal_(self.bos_patch, std=0.02)
        self.patcher, self.theta = patcher, theta        # target-side patching at generation

    def encode(self, src, src_seg):
        M = int(src_seg.max()) + 1
        patches = self.src_local(src, src_seg, M)
        keep = F.one_hot(src_seg, M) * src.ne(self.pad_id).unsqueeze(-1)
        mask = pad_mask(keep.sum(1).eq(0))               # patches holding no real byte
        return self.encoder(self.patch_pe(patches), mask), mask

    def decode(self, tgt, tgt_seg, memory, cross_mask):
        M = int(tgt_seg.max()) + 1
        patches = self.tgt_local(tgt, tgt_seg, M)
        dec_in = torch.cat([self.bos_patch.expand(patches.size(0), -1, -1), patches[:, :-1]], 1)
        z = self.decoder(self.patch_pe(dec_in), memory,
                         causal_mask(dec_in.size(1), dec_in.device), cross_mask)
        bos = torch.full_like(tgt[:, :1], self.bos_id)
        return self.local_decoder(torch.cat([bos, tgt[:, :-1]], 1), tgt_seg, z)

    def forward(self, src, src_seg, tgt, tgt_seg):
        memory, cross_mask = self.encode(src, src_seg)
        return self.decode(tgt, tgt_seg, memory, cross_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_seg, max_len):
        memory, cross_mask = self.encode(src, src_seg)
        B, dev = src.size(0), src.device
        ys = torch.full((B, max_len), self.pad_id, dtype=torch.long, device=dev)
        seg = torch.zeros((B, max_len), dtype=torch.long, device=dev)
        hist = np.full((B, max_len), SEP, dtype=np.int64)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for i in range(max_len):
            if i > 0:   # the boundary at position i depends only on bytes already generated
                prev2 = hist[:, i - 2] if i > 1 else np.full(B, SEP)
                new = self.patcher.entropy_at(prev2, hist[:, i - 1]) > self.theta
                seg[:, i:] = (seg[:, i - 1] + torch.from_numpy(new).long().to(dev)).unsqueeze(1)
            nxt = self.decode(ys, seg, memory, cross_mask)[:, i].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys[:, i] = nxt
            hist[:, i] = nxt.cpu().numpy()
            done |= nxt.eq(self.eos_id)
            if bool(done.all()):
                break
        return ys
