"""Data pipeline: cipher (binary string) -> plaintext, tokenized (C1-C4) or token-free (C5)."""
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from models.blt import ByteEntropyModel
from tokenizer import BPETokenizer, pretokenize_bits, pretokenize_text

CHUNK_CHARS = 64          # plaintext characters per example (8 cipher bits each -> 512 bits)
BITS_PER_CHAR = 8

# byte-level (BLT) vocabulary: 0..255 raw bytes + specials
BYTE_PAD, BYTE_BOS, BYTE_EOS, BYTE_VOCAB = 256, 257, 258, 259


def read_pairs(data_dir: str) -> List[Tuple[str, str]]:
    with open(os.path.join(data_dir, "brown_cipher.txt"), encoding="utf-8") as f:
        cipher = [l for l in f.read().split("\n") if l]
    with open(os.path.join(data_dir, "brown_plain.txt"), encoding="utf-8") as f:
        plain = [l for l in f.read().split("\n") if l]
    assert len(cipher) == len(plain), "files are not line aligned"
    return list(zip(cipher, plain))


def chunk_pairs(pairs, chunk_chars: int = CHUNK_CHARS, min_chars: int = 8):
    """Split each long line into aligned (bits, text) chunks the model can fit."""
    out = []
    for c, p in pairs:
        assert len(c) == BITS_PER_CHAR * len(p), "cipher/plain length mismatch"
        for i in range(0, len(p), chunk_chars):
            text = p[i:i + chunk_chars]
            if len(text) >= min_chars:
                out.append((c[BITS_PER_CHAR * i: BITS_PER_CHAR * (i + len(text))], text))
    return out


def split_by_line(pairs, val_frac=0.05, test_frac=0.05, seed=0):
    """Split on whole lines so no sentence appears in two splits."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pairs), generator=g).tolist()
    n_test, n_val = int(len(pairs) * test_frac), int(len(pairs) * val_frac)
    take = lambda ids: [pairs[i] for i in ids]
    return take(idx[n_test + n_val:]), take(idx[:n_val]), take(idx[n_val:n_val + n_test])


def bits_to_bytes(bits: str) -> List[int]:
    """Group every 8 bits of the ciphertext into one byte value 0-255 (BLT input)."""
    a = np.frombuffer(bits.encode(), dtype=np.uint8) - ord("0")
    return (a.reshape(-1, BITS_PER_CHAR) @ (1 << np.arange(7, -1, -1))).tolist()


# --------------------------------------------------------------------------- tokenized (C1-C4)
def build_tokenizer(train_chunks, path: str, vocab_size: int):
    """Byte-level BPE (own implementation) shared by the binary and the text side."""
    if os.path.exists(path):
        return BPETokenizer.load(path)
    from collections import Counter
    words = Counter()
    for c, p in train_chunks:
        words.update(pretokenize_bits(c))
        words.update(pretokenize_text(p))
    tok = BPETokenizer.train(words, vocab_size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return tok


class TokenizedDataset(Dataset):
    def __init__(self, chunks, tokenizer, max_src: int, max_tgt: int):
        self.src = [tokenizer.encode_bits(c)[:max_src] for c, _ in chunks]
        self.tgt = [tokenizer.encode_text(p)[:max_tgt - 2] for _, p in chunks]
        self.text = [p for _, p in chunks]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i], self.text[i]


def collate_tokenized(batch, pad_id: int, bos_id: int, eos_id: int):
    srcs, tgts, texts = zip(*batch)
    ls, lt = max(len(s) for s in srcs), max(len(t) for t in tgts) + 1
    src = torch.full((len(batch), ls), pad_id, dtype=torch.long)
    tgt_in = torch.full((len(batch), lt), pad_id, dtype=torch.long)
    tgt_out = torch.full((len(batch), lt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, :len(s)] = torch.tensor(s)
        tgt_in[i, :len(t) + 1] = torch.tensor([bos_id] + list(t))
        tgt_out[i, :len(t) + 1] = torch.tensor(list(t) + [eos_id])
    return {"src": src, "tgt_in": tgt_in, "tgt_out": tgt_out}, list(texts)


# --------------------------------------------------------------------------- token-free (C5)
def build_patchers(train_chunks, target_patch: int, sample: int = 4000):
    """Fit the two next-byte entropy models (cipher side, plaintext side) and their thresholds."""
    src_seqs = [bits_to_bytes(c) for c, _ in train_chunks]
    tgt_seqs = [list(p.encode()) + [BYTE_EOS] for _, p in train_chunks]
    out = []
    for seqs in (src_seqs, tgt_seqs):
        m = ByteEntropyModel().fit(seqs)
        out.append((m, m.calibrate(seqs[:sample], target_patch)))
    return out


class ByteDataset(Dataset):
    """Cipher bytes (8 bits -> one byte value) and plaintext bytes, both dynamically patched."""

    def __init__(self, chunks, patchers):
        (sm, sth), (tm, tth) = patchers
        self.src = [bits_to_bytes(c) for c, _ in chunks]
        self.tgt = [list(p.encode()) + [BYTE_EOS] for _, p in chunks]
        self.src_seg = [sm.segments(s, sth) for s in self.src]
        self.tgt_seg = [tm.segments(t, tth) for t in self.tgt]
        self.text = [p for _, p in chunks]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.src_seg[i], self.tgt[i], self.tgt_seg[i], self.text[i]


def collate_bytes(batch):
    srcs, ssegs, tgts, tsegs, texts = zip(*batch)
    B = len(batch)
    ls, lt = max(len(s) for s in srcs), max(len(t) for t in tgts)
    src = torch.full((B, ls), BYTE_PAD, dtype=torch.long)
    tgt = torch.full((B, lt), BYTE_PAD, dtype=torch.long)
    src_seg = torch.zeros((B, ls), dtype=torch.long)
    tgt_seg = torch.zeros((B, lt), dtype=torch.long)
    for i, (s, ss, t, ts) in enumerate(zip(srcs, ssegs, tgts, tsegs)):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
        # padding bytes go into one extra patch of their own, masked out later
        src_seg[i] = torch.tensor(np.concatenate([ss, np.full(ls - len(ss), ss[-1] + 1)]))
        tgt_seg[i] = torch.tensor(np.concatenate([ts, np.full(lt - len(ts), ts[-1] + 1)]))
    return {"src": src, "src_seg": src_seg, "tgt": tgt, "tgt_seg": tgt_seg, "tgt_out": tgt}, \
        list(texts)
