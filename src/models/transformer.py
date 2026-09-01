"""
transformer.py -- the encoder/decoder assembly.

Building blocks:
    FeedForward   : Linear -> GELU -> Linear, position-wise
    EncoderLayer  : pre-norm [ self-attn ] + pre-norm [ FFN ]
    DecoderLayer  : pre-norm [ masked self-attn ] + [ cross-attn ] + [ FFN ]
    Encoder / Decoder : embed (+sinusoidal PE) -> N layers -> final norm
    Seq2SeqTransformer : ties the two, plus greedy generate()

Everything is driven by a `cfg` object (see config.py). Required attributes:
    d_model, n_layers, n_heads, n_kv_heads, d_ff, dropout
    attention     in {"mha", "gqa"}
    norm          in {"layernorm", "rmsnorm"}
    pos_encoding  in {"sinusoidal", "rope"}
    max_src_len, max_tgt_len
    src_vocab_size, tgt_vocab_size
    pad_id, bos_id, eos_id           (assumed shared between the two tokenizers)

Pre-norm everywhere: norm sits *inside* the residual branch, and a final norm
closes each stack. It trains more stably than post-norm at this depth and is
what modern stacks use. The C5 / BLT wrapper reuses Encoder/Decoder via the
`src_emb` / `tgt_emb` hooks (feed patch vectors, skip the token embedding) and
`project=False` (return hidden states for the local byte decoder).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention, MultiHeadAttention
from .norm import LayerNorm, RMSNorm
from .positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding


# --- factories: the only places a Config switch changes the graph ---

def _make_attention(cfg, rope: Optional[nn.Module]) -> nn.Module:
    if cfg.attention == "gqa":
        return GroupedQueryAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout, rope=rope
        )
    return MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, rope=rope)


def _make_norm(cfg, dim: int) -> nn.Module:
    return RMSNorm(dim) if cfg.norm == "rmsnorm" else LayerNorm(dim)


class FeedForward(nn.Module):
    """Position-wise FFN: widen to d_ff, GELU, project back. Applied identically
    at every position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, cfg, rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.norm1 = _make_norm(cfg, cfg.d_model)
        self.attn = _make_attention(cfg, rope=rope)          # self-attn carries RoPE
        self.norm2 = _make_norm(cfg, cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h, self_mask))        # pre-norm self-attn
        x = x + self.drop(self.ffn(self.norm2(x)))           # pre-norm FFN
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg, rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.norm1 = _make_norm(cfg, cfg.d_model)
        self.self_attn = _make_attention(cfg, rope=rope)     # masked self-attn carries RoPE
        self.norm2 = _make_norm(cfg, cfg.d_model)
        self.cross_attn = _make_attention(cfg, rope=None)    # cross-attn: query/key live in
        self.norm3 = _make_norm(cfg, cfg.d_model)            #   different sequences -> no RoPE
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, self_mask))
        h = self.norm2(x)
        x = x + self.drop(self.cross_attn(h, memory, cross_mask))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg, rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        # sinusoidal path adds PE here; rope path adds nothing (rotation happens in attn)
        self.pos = (
            SinusoidalPositionalEncoding(cfg.d_model, cfg.max_src_len, dropout=0.0)
            if cfg.pos_encoding == "sinusoidal"
            else None
        )
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg, rope) for _ in range(cfg.n_layers))
        self.norm_out = _make_norm(cfg, cfg.d_model)

    def forward(
        self,
        src_ids: Optional[torch.Tensor] = None,
        src_emb: Optional[torch.Tensor] = None,   # BLT hook: pass patch vectors, skip embed
        self_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_emb is None:
            src_emb = self.embed(src_ids) * math.sqrt(self.d_model)   # Vaswani-style scale
            if self.pos is not None:
                src_emb = self.pos(src_emb)
        x = self.emb_drop(src_emb)
        for layer in self.layers:
            x = layer(x, self_mask)
        return self.norm_out(x)


class Decoder(nn.Module):
    def __init__(self, cfg, rope: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = (
            SinusoidalPositionalEncoding(cfg.d_model, cfg.max_tgt_len, dropout=0.0)
            if cfg.pos_encoding == "sinusoidal"
            else None
        )
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(DecoderLayer(cfg, rope) for _ in range(cfg.n_layers))
        self.norm_out = _make_norm(cfg, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        self.out_proj.weight = self.embed.weight                      # weight tying

    def forward(
        self,
        tgt_ids: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        tgt_emb: Optional[torch.Tensor] = None,   # BLT hook
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        project: bool = True,                      # BLT sets False -> return hidden states
    ) -> torch.Tensor:
        if tgt_emb is None:
            tgt_emb = self.embed(tgt_ids) * math.sqrt(self.d_model)
            if self.pos is not None:
                tgt_emb = self.pos(tgt_emb)
        x = self.emb_drop(tgt_emb)
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        x = self.norm_out(x)
        return self.out_proj(x) if project else x


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder. `forward` is teacher-forced (feed the gold prefix);
    `generate` is greedy autoregressive decoding."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # One RoPE instance shared by every self-attention block: it holds only
        # constant buffers, and encoder/decoder self-attn both index positions
        # from 0, so sharing is safe and saves the cos/sin tables.
        rope = (
            RotaryPositionalEmbedding(
                cfg.d_model // cfg.n_heads,
                max_len=max(cfg.max_src_len, cfg.max_tgt_len),
            )
            if cfg.pos_encoding == "rope"
            else None
        )
        self.encoder = Encoder(cfg, rope)
        self.decoder = Decoder(cfg, rope)
        self.pad_id, self.bos_id, self.eos_id = cfg.pad_id, cfg.bos_id, cfg.eos_id

        # --- weight init ---
        # Linears: xavier_uniform. Embeddings: N(0, d_model^-0.5) so that after
        # the forward's *sqrt(d_model) scale they have std ~1, on par with the
        # O(1) sinusoidal PE (Vaswani convention). Without this, tied-embedding
        # logits blow up and the untrained loss is ~vocab, not ~ln(vocab).
        self.apply(self._init_linear)
        for emb in (self.encoder.embed, self.decoder.embed):
            nn.init.normal_(emb.weight, std=cfg.d_model ** -0.5)
            with torch.no_grad():
                emb.weight[cfg.pad_id].zero_()
        self.decoder.out_proj.weight = self.decoder.embed.weight   # keep tied

    @staticmethod
    def _init_linear(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # --- masks: bool, broadcastable to (B, H, Tq, Tk), True = "may attend".
    #     Kept here for a self-contained module; could move to utils.py. ---
    def _pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids != self.pad_id)[:, None, None, :]              # (B, 1, 1, T)

    @staticmethod
    def _causal_mask(t: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))[None, None]  # (1,1,T,T)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_kpm = self._pad_mask(src)                              # keys = source tokens
        tgt_kpm = self._pad_mask(tgt_in)
        causal = self._causal_mask(tgt_in.size(1), tgt_in.device)
        memory = self.encoder(src_ids=src, self_mask=src_kpm)
        logits = self.decoder(
            tgt_ids=tgt_in,
            memory=memory,
            self_mask=causal & tgt_kpm,                            # (B, 1, Tt, Tt)
            cross_mask=src_kpm,                                    # (B, 1, 1, Ts)
        )
        return logits                                             # (B, Tt, tgt_vocab)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int) -> torch.Tensor:
        """Greedy decode. Recomputes the whole prefix each step (no KV cache) --
        simple and fine at this sequence scale."""
        was_training = self.training
        self.eval()
        src_kpm = self._pad_mask(src)
        memory = self.encoder(src_ids=src, self_mask=src_kpm)

        b = src.size(0)
        ys = torch.full((b, 1), self.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(b, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            causal = self._causal_mask(ys.size(1), src.device)
            logits = self.decoder(
                tgt_ids=ys, memory=memory, self_mask=causal, cross_mask=src_kpm
            )
            nxt = logits[:, -1].argmax(dim=-1)                     # greedy pick
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= nxt == self.eos_id
            if bool(done.all()):
                break

        if was_training:
            self.train()
        return ys
