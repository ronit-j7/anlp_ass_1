"""
blt.py -- simplified Byte Latent Transformer (Configuration 5).

No vocabulary. The pipeline:

    raw bytes --local encoder--> patch vectors --global transformer--> patch
    hidden states --local decoder--> raw byte logits

Simplifications vs the paper (Pagnoni et al. 2024):
  * fixed patch size P (no entropy-based dynamic patching)
  * the local encoder pools each P-byte block independently (within-patch
    attention only), so training and single-patch generation are identical ops
  * the local decoder predicts a patch's P bytes in parallel from the global
    hidden state (no autoregression *within* a patch); the global decoder is
    still autoregressive *over* patches

The global encoder/decoder are the same modules used by C1-C4 (transformer.py),
fed patch vectors through their src_emb / tgt_emb hooks with project=False.

Interface matches Seq2SeqTransformer so train.py / eval treat it the same:
  forward(src_bytes, tgt_bytes) -> (byte_logits, labels)   # BLT returns the pair
  generate(src_bytes, max_len)  -> (B, L) byte ids, BOS-prefixed
Byte ids: 0..3 are <pad>/<bos>/<eos>/<unk>, byte value v is id v+4 (vocab 260).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding
from .transformer import Decoder, Encoder, FeedForward

BYTE_VOCAB = 260  # 4 specials + 256 byte values


class LocalEncoder(nn.Module):
    """Embed P bytes of a patch, mix them with within-patch self-attention,
    then pool to one vector. Operates on (B, N, P) byte ids -> (B, N, d)."""

    def __init__(self, d_model: int, patch_size: int, n_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Embedding(BYTE_VOCAB, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, patch_size, d_model) * 0.02)
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, 4 * d_model, dropout)
        self.pool = nn.Linear(patch_size * d_model, d_model)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        b, n, p = byte_ids.shape
        x = self.embed(byte_ids) + self.pos                 # (B, N, P, d)
        x = x.reshape(b * n, p, -1)
        h = self.norm1(x)
        x = x + self.attn(h, h)                             # within-patch attention, no mask
        x = x + self.ffn(self.norm2(x))
        x = x.reshape(b, n, p * x.size(-1))
        return self.pool(x)                                 # (B, N, d)


class LocalDecoder(nn.Module):
    """Expand each patch hidden state back to P byte logits (parallel over the
    P positions). (B, N, d) -> (B, N*P, BYTE_VOCAB)."""

    def __init__(self, d_model: int, patch_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.up = nn.Linear(d_model, patch_size * d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, patch_size, d_model) * 0.02)
        self.act = nn.GELU()
        self.norm = LayerNorm(d_model)
        self.head = nn.Linear(d_model, BYTE_VOCAB)

    def forward(self, patch_h: torch.Tensor) -> torch.Tensor:
        b, n, d = patch_h.shape
        x = self.up(patch_h).view(b, n, self.patch_size, d) + self.pos
        x = self.norm(self.act(x))
        return self.head(x).view(b, n * self.patch_size, BYTE_VOCAB)


class ByteLatentTransformer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.P = cfg.patch_size
        self.pad_id, self.bos_id, self.eos_id = cfg.pad_id, cfg.bos_id, cfg.eos_id
        d = cfg.d_model
        max_patches = cfg.max_tgt_len // self.P + 4

        # separate local encoders for cipher-bytes and plaintext-bytes
        self.src_local = LocalEncoder(d, self.P, cfg.n_heads, cfg.dropout)
        self.tgt_local = LocalEncoder(d, self.P, cfg.n_heads, cfg.dropout)
        self.enc_pos = SinusoidalPositionalEncoding(d, max_patches)
        self.dec_pos = SinusoidalPositionalEncoding(d, max_patches)
        self.global_encoder = Encoder(cfg, rope=None)       # used via src_emb hook
        self.global_decoder = Decoder(cfg, rope=None)       # used via tgt_emb hook, project=False
        self.start_patch = nn.Parameter(torch.zeros(1, 1, d))
        self.local_decoder = LocalDecoder(d, self.P, cfg.dropout)

    # --- helpers ---
    def _pad_to_patch(self, x: torch.Tensor) -> torch.Tensor:
        rem = x.size(1) % self.P
        if rem:
            x = torch.cat([x, x.new_full((x.size(0), self.P - rem), self.pad_id)], dim=1)
        return x

    def _patchify(self, byte_ids: torch.Tensor, local: LocalEncoder):
        byte_ids = self._pad_to_patch(byte_ids)
        b, l = byte_ids.shape
        blocks = byte_ids.view(b, l // self.P, self.P)              # (B, N, P)
        patches = local(blocks)                                     # (B, N, d)
        patch_real = (blocks != self.pad_id).any(dim=-1)            # (B, N) bool
        return patches, patch_real, byte_ids

    @staticmethod
    def _causal(n: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))[None, None]

    # --- training ---
    def forward(self, src_bytes: torch.Tensor, tgt_bytes: torch.Tensor):
        src_patches, src_real, _ = self._patchify(src_bytes, self.src_local)
        tgt_patches, tgt_real, tgt_padded = self._patchify(tgt_bytes, self.tgt_local)

        src_mask = src_real[:, None, None, :]                       # (B,1,1,Ns)
        memory = self.global_encoder(src_emb=self.enc_pos(src_patches), self_mask=src_mask)

        # decoder input = start patch + gold patches[:-1]  (patch-level teacher forcing)
        b, n, d = tgt_patches.shape
        dec_in = torch.cat([self.start_patch.expand(b, 1, d), tgt_patches[:, :-1]], dim=1)
        causal = self._causal(n, tgt_patches.device)
        h = self.global_decoder(
            tgt_emb=self.dec_pos(dec_in), memory=memory,
            self_mask=causal & tgt_real[:, None, None, :],
            cross_mask=src_mask, project=False,
        )                                                          # (B, N, d)
        byte_logits = self.local_decoder(h)                        # (B, N*P, V)
        return byte_logits, tgt_padded                             # labels aligned to N*P

    # --- inference (greedy, patch-autoregressive) ---
    @torch.no_grad()
    def generate(self, src_bytes: torch.Tensor, max_len: int) -> torch.Tensor:
        was_training = self.training
        self.eval()
        src_patches, src_real, _ = self._patchify(src_bytes, self.src_local)
        src_mask = src_real[:, None, None, :]
        memory = self.global_encoder(src_emb=self.enc_pos(src_patches), self_mask=src_mask)

        b = src_bytes.size(0)
        d = src_patches.size(-1)
        cur = self.start_patch.expand(b, 1, d).clone()             # (B, 1, d)
        out = []
        done = torch.zeros(b, dtype=torch.bool, device=src_bytes.device)
        for _ in range(max(1, max_len // self.P)):
            n = cur.size(1)
            h = self.global_decoder(
                tgt_emb=self.dec_pos(cur), memory=memory,
                self_mask=self._causal(n, cur.device), cross_mask=src_mask, project=False,
            )
            nb = self.local_decoder(h[:, -1:]).argmax(dim=-1)      # (B, P)
            nb = torch.where(done[:, None], torch.full_like(nb, self.pad_id), nb)
            out.append(nb)
            done |= (nb == self.eos_id).any(dim=1)
            if bool(done.all()):
                break
            next_patch = self.tgt_local(nb[:, None, :])            # (B, 1, d)
            cur = torch.cat([cur, next_patch], dim=1)

        ys = torch.cat(out, dim=1) if out else src_bytes.new_zeros((b, 0))
        bos = ys.new_full((b, 1), self.bos_id)
        if was_training:
            self.train()
        return torch.cat([bos, ys], dim=1)                         # BOS-prefixed, like Seq2Seq
