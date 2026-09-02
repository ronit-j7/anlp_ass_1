"""
config.py -- one Config dataclass and the five ablation configurations.

C1 is the base. C2..C5 are each `replace(C1, <one field>)`, so exactly one
component changes per variant -- the whole point of the ablation, and it keeps
the setup auditable at a glance. Anything a variant does not name is inherited
from C1 (depth, width, dropout, LR, batch size, schedule, seed, ...), which is
the "consistent hyperparameters where applicable" the brief asks for.

src_vocab_size / tgt_vocab_size depend on the trained tokenizers, so they are
None here and filled in by train.py via `dataclasses.replace` once the
tokenizers are built or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

# Special-token ids, fixed identically in the source and target tokenizers so
# the model shares one pad/bos/eos convention.
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass(frozen=True)
class Config:
    name: str = "C1"

    # --- ablation axes: exactly one of these differs from C1 per variant ---
    pos_encoding: str = "sinusoidal"     # "sinusoidal" | "rope"
    attention: str = "mha"               # "mha" | "gqa"
    norm: str = "layernorm"              # "layernorm" | "rmsnorm"
    tokenization: str = "subword"        # "subword" | "blt"

    # --- model width / depth (shared across all configs) ---
    d_model: int = 256                   # ~9M; generalizing the XOR+phase rule needs the capacity
    n_layers: int = 4                    # same count for encoder and decoder
    n_heads: int = 8
    n_kv_heads: int = 2                  # only consulted when attention == "gqa"
    d_ff: int = 1024
    dropout: float = 0.1

    # --- tokenizer / sequence (shared) ---
    src_merge_ops: int = 3000            # byte-level BPE on the cipher (see dataset.py header)
    tgt_merge_ops: int = 4000            # word-level BPE on the plaintext
    patch_size: int = 4                  # BLT only (C5): bytes per patch
    chunk_chars: int = 256              # split each line into windows of this many PLAINTEXT
                                        # chars BEFORE tokenizing (0 = whole line). Must be a
                                        # multiple of 8: then every chunk starts at cipher phase
                                        # 0, so phase == (local position mod 8) -- directly
                                        # available from the positional encoding instead of
                                        # something the model must count across the whole line.
                                        # Also ~2.7x more, shorter training pairs.
    max_src_len: int = 384             # a 256-char chunk BPE-encodes well under this
    max_tgt_len: int = 384             # over-cap chunks: dropped in train, truncated in eval
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    unk_id: int = UNK_ID

    # --- optimization (shared) ---
    batch_size: int = 32                # nominal; real batching is token-budget (see max_tokens)
    max_tokens: int = 12288            # cap on (rows * padded_len) per train/val batch, so a
                                        # long example -> fewer rows. Attention here is O(B*T^2)
                                        # (no flash-attn), so this is what bounds GPU memory.
    eval_batch_size: int = 16          # greedy decode is autoregressive; keep this modest
    lr: float = 5e-4                     # peak LR after warmup
    lr_schedule: str = "cosine"         # "cosine" | "inverse_sqrt"
    lr_min_ratio: float = 0.1          # cosine floor: LR never drops below 10% of peak
    warmup_steps: int = 800
    max_steps: int = 12000             # C1 was still improving at 7k before the LR collapse
    min_steps: int = 4000              # early stopping cannot fire before this
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.0       # deterministic exact-reconstruction task -> want the
                                        # model fully confident; smoothing caps seq_acc
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    adam_eps: float = 1e-9
    bf16: bool = True

    # --- eval / logging (shared) ---
    eval_every: int = 1000
    eval_subset: int = 128              # greedy-decode this many val chunks per eval
    eval_gen_max_len: int = 384        # chunks are short now, so decode them in full
    early_stop_patience: int = 4       # consecutive evals w/o val gain (seq_acc, then loss)
    seed: int = 42
    wandb_project: str = "anlp-a1-transformers"

    # --- data / paths ---
    data_dir: str = "."
    cipher_file: str = "brown_cipher.txt"
    plain_file: str = "brown_plain.txt"
    n_val: int = 250
    n_test: int = 250
    num_workers: int = 0        # dataset is small and in-memory; workers = pure overhead
    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    tokenizer_dir: str = "tokenizers"

    def __post_init__(self) -> None:
        assert self.pos_encoding in ("sinusoidal", "rope")
        assert self.attention in ("mha", "gqa")
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.tokenization in ("subword", "blt")
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide by n_kv_heads"

    @property
    def gen_max_len(self) -> int:
        return self.max_tgt_len


# --- C1 base, then strictly one-field variants ---
C1 = Config(name="C1")
C2 = replace(C1, name="C2", pos_encoding="rope")
C3 = replace(C1, name="C3", attention="gqa")
C4 = replace(C1, name="C4", norm="rmsnorm")
C5 = replace(C1, name="C5", tokenization="blt")   # model/data path wired with blt.py

CONFIGS = {c.name: c for c in (C1, C2, C3, C4, C5)}


def get_config(name: str) -> Config:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose from {list(CONFIGS)}")
    return CONFIGS[name]
