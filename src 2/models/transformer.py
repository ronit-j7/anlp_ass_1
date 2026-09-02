"""Pre-norm encoder-decoder Transformer assembled from the custom modules."""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .attention import FeedForward, GroupedQueryAttention, MultiHeadAttention
from .norm import make_norm
from .positional import RotaryPositionalEmbedding, SinusoidalPositionalEncoding

NEG_INF = -1e9


@dataclass
class ModelConfig:
    src_vocab: int = 0
    tgt_vocab: int = 0
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 8
    n_kv_heads: int = 8          # < n_heads -> GQA
    d_ff: int = 1024
    dropout: float = 0.1
    pos: str = "sinusoidal"      # sinusoidal | rope
    norm: str = "layernorm"      # layernorm | rmsnorm
    max_len: int = 1024
    # BLT only
    n_local_layers: int = 1


def causal_mask(T: int, device) -> torch.Tensor:
    m = torch.full((T, T), NEG_INF, device=device).triu(1)
    return m.view(1, 1, T, T)


def pad_mask(is_pad: torch.Tensor) -> torch.Tensor:
    """is_pad: (B, Lk) bool -> additive mask (B, 1, 1, Lk)."""
    return is_pad[:, None, None, :].float() * NEG_INF


def init_weights(model: nn.Module, d_model: int):
    """Xavier for projections, N(0, 1/sqrt(d_model)) for embeddings (unit scale after * sqrt(d))."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=d_model ** -0.5)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)


def _attn(cfg: ModelConfig, rope) -> nn.Module:
    if cfg.n_kv_heads < cfg.n_heads:
        return GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout, rope)
    return MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.dropout, rope)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, rope=None):
        super().__init__()
        self.norm1, self.norm2 = make_norm(cfg.norm, cfg.d_model), make_norm(cfg.norm, cfg.d_model)
        self.self_attn = _attn(cfg, rope)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, rope=None):
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.d_model)
        self.norm2 = make_norm(cfg.norm, cfg.d_model)
        self.norm3 = make_norm(cfg.norm, cfg.d_model)
        self.self_attn = _attn(cfg, rope)
        self.cross_attn = _attn(cfg, None)      # no RoPE across two different sequences
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, self_mask))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, cross_mask))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    """Stack of pre-norm encoder layers operating on already-embedded inputs."""

    def __init__(self, cfg: ModelConfig, rope=None):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg, rope) for _ in range(cfg.n_layers)])
        self.norm = make_norm(cfg.norm, cfg.d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig, rope=None):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg, rope) for _ in range(cfg.n_layers)])
        self.norm = make_norm(cfg.norm, cfg.d_model)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)


class Seq2SeqTransformer(nn.Module):
    """Tokenized (subword) encoder-decoder used by configurations C1-C4."""

    ENCODE_KEYS = ("src",)

    def __init__(self, cfg: ModelConfig, pad_id: int, bos_id: int, eos_id: int):
        super().__init__()
        self.cfg, self.pad_id, self.bos_id, self.eos_id = cfg, pad_id, bos_id, eos_id
        rope = None
        self.pe = None
        if cfg.pos == "rope":
            rope = RotaryPositionalEmbedding(cfg.d_model // cfg.n_heads, cfg.max_len)
        else:
            self.pe = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.src_emb = nn.Embedding(cfg.src_vocab, cfg.d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab, cfg.d_model, padding_idx=pad_id)
        self.encoder = Encoder(cfg, rope)
        self.decoder = Decoder(cfg, rope)
        self.lm_head = nn.Linear(cfg.d_model, cfg.tgt_vocab, bias=False)
        self.lm_head.weight = self.tgt_emb.weight            # weight tying
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = cfg.d_model ** 0.5
        init_weights(self, cfg.d_model)
        self.lm_head.weight = self.tgt_emb.weight            # keep tying after init

    def _embed(self, ids, emb):
        x = emb(ids) * self.scale
        return self.pe(x) if self.pe is not None else self.drop(x)

    def encode(self, src):
        src_pad = src.eq(self.pad_id)
        memory = self.encoder(self._embed(src, self.src_emb), pad_mask(src_pad))
        return memory, pad_mask(src_pad)

    def decode(self, tgt_in, memory, cross_mask):
        x = self._embed(tgt_in, self.tgt_emb)
        self_mask = causal_mask(tgt_in.size(1), tgt_in.device)
        return self.lm_head(self.decoder(x, memory, self_mask, cross_mask))

    def forward(self, src, tgt_in):
        memory, cross_mask = self.encode(src)
        return self.decode(tgt_in, memory, cross_mask)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int):
        memory, cross_mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, memory, cross_mask)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt.eq(self.eos_id)
            if bool(done.all()):
                break
        return ys[:, 1:]
