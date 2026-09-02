"""
blt.py -- simplified Byte Latent Transformer (Configuration 5).

Pipeline (no vocabulary):

    bytes --local encoder--> patch vectors --global transformer--> patch
    hidden states --local decoder--> byte logits

Key BLT idea kept: **entropy-based dynamic patching**. A lightweight order-1
byte Markov model (ByteEntropyModel) estimates H(next byte | prev byte); a new
patch starts wherever that entropy exceeds a threshold, so predictable byte
runs become one patch and unpredictable regions get finer patches. No separate
neural entropy model, no fixed-width patches.

Simplifications vs the paper:
  * order-1 Markov entropy estimator instead of a byte-level SLM
  * local decoder predicts a patch's bytes in parallel from the global hidden
    (no autoregression *within* a patch); the global decoder is still
    autoregressive *over* patches, and at generation the entropy model is run
    on the emitted bytes to decide where each patch ends
  * a learned 256-entry byte embedding (per side)

Byte ids: value 0..255 = that byte; id 256 = <pad>. No BOS/EOS -- a chunk's
target byte count equals its source byte count, so decoding length is known.
"""

from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding
from .transformer import Decoder, Encoder, FeedForward

BYTE_PAD = 256
BYTE_VOCAB = 257            # 256 byte values + <pad>


# =============================================================================
# lightweight entropy estimator for dynamic patching
# =============================================================================
class ByteEntropyModel:
    """Order-1 byte Markov model. H1[b] = Shannon entropy (bits) of the next
    byte given that the previous byte is b. `theta` is chosen so the mean
    patch length over the fitting corpus is ~`target_mean_patch`."""

    def __init__(self, smoothing: float = 0.5):
        self.smoothing = smoothing
        self.H1 = np.zeros(256, dtype=np.float64)
        self.H0 = 0.0
        self.theta = 0.0
        self.max_patch = 12

    def fit(self, byte_seqs: List[List[int]], target_mean_patch: float = 5.0,
            max_patch: int = 12) -> "ByteEntropyModel":
        self.max_patch = max_patch
        c1 = np.zeros((256, 256), dtype=np.float64)
        c0 = np.zeros(256, dtype=np.float64)
        for s in byte_seqs:
            a = np.asarray(s, dtype=np.int64)
            a = a[(a >= 0) & (a < 256)]
            if a.size == 0:
                continue
            np.add.at(c0, a, 1.0)
            if a.size > 1:
                np.add.at(c1, (a[:-1], a[1:]), 1.0)
        p1 = c1 + self.smoothing
        p1 /= p1.sum(axis=1, keepdims=True)
        self.H1 = -(p1 * np.log2(p1)).sum(axis=1)                 # (256,)
        p0 = c0 + self.smoothing
        p0 /= p0.sum()
        self.H0 = float(-(p0 * np.log2(p0)).sum())

        ent = []
        for s in byte_seqs:
            a = np.asarray(s, dtype=np.int64)
            a = a[(a >= 0) & (a < 256)]
            if a.size > 1:
                ent.append(self.H1[a[:-1]])
        ent = np.concatenate(ent) if ent else np.array([self.H0])
        q = max(0.0, min(1.0, 1.0 - 1.0 / max(target_mean_patch, 1.0)))
        self.theta = float(np.quantile(ent, q))
        return self

    def patch_lengths(self, byte_vals: List[int]) -> List[int]:
        """Greedy: a boundary sits *before* position t (t>=1) when
        H(next | byte[t-1]) > theta; also force one at max_patch."""
        n = len(byte_vals)
        if n == 0:
            return []
        lens, cur = [], 1
        for t in range(1, n):
            prev = byte_vals[t - 1]
            h = self.H1[prev] if 0 <= prev < 256 else self.H0
            if h > self.theta or cur >= self.max_patch:
                lens.append(cur)
                cur = 1
            else:
                cur += 1
        lens.append(cur)
        return lens

    # torch view of H1, for the generation-time cut
    def h_of(self, byte_val: int) -> float:
        return float(self.H1[byte_val]) if 0 <= byte_val < 256 else self.H0

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"H1": self.H1, "H0": self.H0, "theta": self.theta,
                         "max_patch": self.max_patch}, f)

    @classmethod
    def load(cls, path: str) -> "ByteEntropyModel":
        m = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        m.H1, m.H0, m.theta, m.max_patch = d["H1"], d["H0"], d["theta"], d["max_patch"]
        return m


# =============================================================================
# patch <-> byte segmentation (given per-patch lengths)
# =============================================================================
def segment_bytes(byte_ids: torch.Tensor, plen: torch.Tensor, max_patch: int):
    """byte_ids (B, L), plen (B, N) with 0 padding -> gathered (B, N, max_patch)
    byte ids, a (B, N, max_patch) validity mask, and a (B, N) patch mask."""
    b, l = byte_ids.shape
    n = plen.size(1)
    starts = torch.cat([plen.new_zeros(b, 1), plen.cumsum(1)[:, :-1]], dim=1)   # (B, N)
    ar = torch.arange(max_patch, device=byte_ids.device)
    idx = (starts[:, :, None] + ar[None, None, :]).clamp_(max=l - 1)            # (B, N, P)
    valid = (ar[None, None, :] < plen[:, :, None]) & (plen[:, :, None] > 0)
    g = torch.gather(byte_ids[:, None, :].expand(b, n, l), 2, idx)
    g = torch.where(valid, g, g.new_full((), BYTE_PAD))
    return g, valid, plen > 0


# =============================================================================
# local encoder / decoder
# =============================================================================
class LocalEncoder(nn.Module):
    """Embed a patch's bytes, mix with within-patch attention, pool to a vector.
    (B, N, max_patch) byte ids -> (B, N, d)."""

    def __init__(self, d_model: int, max_patch: int, n_heads: int, dropout: float):
        super().__init__()
        self.max_patch = max_patch
        self.embed = nn.Embedding(BYTE_VOCAB, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, max_patch, d_model) * 0.02)
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4 * d_model, dropout)
        self.pool = nn.Linear(max_patch * d_model, d_model)

    def forward(self, patch_bytes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, n, p = patch_bytes.shape
        x = self.embed(patch_bytes) + self.pos[:, :, :p]              # (B, N, P, d)
        x = x.reshape(b * n, p, -1)
        m = valid.reshape(b * n, 1, 1, p)                            # keep-mask
        h = self.norm1(x)
        x = x + self.attn(h, h, m)
        x = x + self.ffn(self.norm2(x))
        return self.pool(x.reshape(b, n, p * x.size(-1)))            # (B, N, d)


class LocalDecoder(nn.Module):
    """Patch hidden state -> the patch's bytes, predicted in parallel.
    (B, N, d) -> (B, N, max_patch, BYTE_VOCAB)."""

    def __init__(self, d_model: int, max_patch: int):
        super().__init__()
        self.max_patch = max_patch
        self.up = nn.Linear(d_model, max_patch * d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, max_patch, d_model) * 0.02)
        self.act = nn.GELU()
        self.norm = LayerNorm(d_model)
        self.head = nn.Linear(d_model, BYTE_VOCAB)

    def forward(self, patch_h: torch.Tensor) -> torch.Tensor:
        b, n, d = patch_h.shape
        x = self.up(patch_h).view(b, n, self.max_patch, d) + self.pos
        return self.head(self.norm(self.act(x)))                    # (B, N, P, V)


# =============================================================================
# the C5 model
# =============================================================================
class ByteLatentTransformer(nn.Module):
    def __init__(self, cfg, src_entropy: ByteEntropyModel, tgt_entropy: ByteEntropyModel):
        super().__init__()
        self.cfg = cfg
        self.src_entropy = src_entropy
        self.tgt_entropy = tgt_entropy
        self.P = max(src_entropy.max_patch, tgt_entropy.max_patch)
        d = cfg.d_model
        max_patches = cfg.max_tgt_len + 4                            # loose upper bound

        self.src_local = LocalEncoder(d, self.P, cfg.n_heads, cfg.dropout)
        self.tgt_local = LocalEncoder(d, self.P, cfg.n_heads, cfg.dropout)
        self.enc_pos = SinusoidalPositionalEncoding(d, max_patches)
        self.dec_pos = SinusoidalPositionalEncoding(d, max_patches)
        self.global_encoder = Encoder(cfg, rope=None)                # via src_emb hook
        self.global_decoder = Decoder(cfg, rope=None)                # via tgt_emb hook, project=False
        self.start_patch = nn.Parameter(torch.zeros(1, 1, d))
        self.local_decoder = LocalDecoder(d, self.P)

    @staticmethod
    def _causal(n: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))[None, None]

    # --- training: byte-aligned logits + labels (labels already -100 on pad slots) ---
    def forward(self, src, tgt, src_plen, tgt_plen):
        s_bytes, s_valid, s_pmask = segment_bytes(src, src_plen, self.P)
        t_bytes, t_valid, t_pmask = segment_bytes(tgt, tgt_plen, self.P)

        s_patches = self.src_local(s_bytes, s_valid)                 # (B, Ns, d)
        t_patches = self.tgt_local(t_bytes, t_valid)                 # (B, Nt, d)

        s_mask = s_pmask[:, None, None, :]
        memory = self.global_encoder(src_emb=self.enc_pos(s_patches), self_mask=s_mask)

        b, n, d = t_patches.shape
        dec_in = torch.cat([self.start_patch.expand(b, 1, d), t_patches[:, :-1]], dim=1)
        h = self.global_decoder(
            tgt_emb=self.dec_pos(dec_in), memory=memory,
            self_mask=self._causal(n, t_patches.device) & t_pmask[:, None, None, :],
            cross_mask=s_mask, project=False,
        )                                                           # (B, Nt, d)
        logits = self.local_decoder(h)                              # (B, Nt, P, V)
        labels = torch.where(t_valid, t_bytes, t_bytes.new_full((), -100))
        return logits.reshape(b, n * self.P, -1), labels.reshape(b, n * self.P)

    # --- inference: patch-autoregressive; entropy model cuts each emitted patch ---
    @torch.no_grad()
    def generate(self, src, src_plen, max_len: int) -> torch.Tensor:
        was_training = self.training
        self.eval()
        s_bytes, s_valid, s_pmask = segment_bytes(src, src_plen, self.P)
        s_patches = self.src_local(s_bytes, s_valid)
        s_mask = s_pmask[:, None, None, :]
        memory = self.global_encoder(src_emb=self.enc_pos(s_patches), self_mask=s_mask)

        b, d = s_patches.size(0), s_patches.size(-1)
        target_len = (src != BYTE_PAD).sum(dim=1).tolist()          # tgt byte count == src byte count
        cur = self.start_patch.expand(b, 1, d).clone()
        rows: List[List[int]] = [[] for _ in range(b)]
        te = self.tgt_entropy

        for _ in range(max(1, max_len)):
            n = cur.size(1)
            h = self.global_decoder(
                tgt_emb=self.dec_pos(cur), memory=memory,
                self_mask=self._causal(n, cur.device), cross_mask=s_mask, project=False,
            )
            cand = self.local_decoder(h[:, -1:]).argmax(dim=-1)[:, 0].tolist()   # (B, P)

            nb = torch.full((b, self.P), BYTE_PAD, dtype=torch.long, device=src.device)
            plen = torch.ones(b, dtype=torch.long, device=src.device)
            for j in range(b):
                rem = target_len[j] - len(rows[j])
                if rem <= 0:
                    continue
                lim = min(self.P, rem)
                take = 1
                for t in range(1, lim):
                    if te.h_of(cand[j][t - 1]) > te.theta or take >= te.max_patch:
                        break
                    take += 1
                for k in range(take):
                    nb[j, k] = cand[j][k]
                    rows[j].append(cand[j][k])
                plen[j] = take

            if all(len(rows[j]) >= target_len[j] for j in range(b)):
                break
            pb, pv, _ = segment_bytes(nb, plen[:, None], self.P)
            cur = torch.cat([cur, self.tgt_local(pb, pv)], dim=1)

        m = max((target_len[j] for j in range(b)), default=0)
        ys = torch.full((b, max(m, 1)), BYTE_PAD, dtype=torch.long, device=src.device)
        for j in range(b):
            r = rows[j][: target_len[j]]
            if r:
                ys[j, : len(r)] = torch.tensor(r, device=src.device)
        return ys
