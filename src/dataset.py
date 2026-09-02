"""
dataset.py -- tokenizers, encoded datasets, and collation for C1-C4.

Task direction: SOURCE = cipher (binary string), TARGET = plaintext English.
The model reads bits and generates text.

------------------------------------------------------------------------------
Why the cipher side uses BYTE-level base units (and not bit-level)
------------------------------------------------------------------------------
We first tried true bit-level BPE: base alphabet {0, 1}, each cipher line as
one segment, merges allowed anywhere. It is not viable with this trainer.

    Benchmark: 1,500 cipher lines, 2,000 merges  ->  ~2.4 s/merge and rising,
    ~1.5 h projected -- at 1/3 of the data and 1/3 of the target merges, and
    before any encoding.

Root cause (algorithmic, not a tuning knob): BPETokenizer is word-*type*
based. On English it is fast because ~30k word types with Zipfian frequency
mean a merge only rewrites the few short types that contain the pair. The
cipher has no words: one segment per line, every line unique (zero dedup),
~4,780 symbols each. So every common bit-pair ("01", "10", ...) touches
*every* line and rescans all ~4,780 symbols per merge.

Fix: base unit = one byte (8 bits). Segments drop ~8x (4,780 -> ~600 symbols)
and BPE then merges frequent byte / multi-byte runs on top. This is exactly
GPT-2 / LLaMA "byte-level BPE" -- still a standard subword tokenizer. Byte
alignment is a property of how text is written, not the cipher's secret; the
model still has to learn the position-mod-8 XOR mapping between cipher
subwords and plaintext subwords (key = "ANLP2026"), which is the real task.

C5 (BLT) is the token-free contrast and is handled in blt.py.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .config import BOS_ID, EOS_ID, PAD_ID, SPECIALS, UNK_ID, Config

try:
    from tqdm import tqdm
except ImportError:                       # tqdm is optional
    def tqdm(x, **_):
        return x

# BPE merges generalize fine from a sample; cap the cipher training corpus so
# merge learning stays bounded (it has no word-type dedup -- see header). With
# 1500 lines / 3000 merges this is a ~20-25 min ONE-TIME cost, then the
# tokenizer is pickled and every run loads it instantly.
_CIPHER_BPE_TRAIN_LINES = 1500
_BYTE = 8


# =============================================================================
# BPETokenizer -- the provided incremental trainer, lightly generalized so the
# same class serves both sides:
#   * pretokenizer : line -> list[segment]   (merges never cross a segment)
#   * symbolizer   : segment -> list[base symbol]
#   * end_of_word=None disables the word-boundary marker (used for the cipher)
# Changes from the original are marked  # [gen].
# =============================================================================
class BPETokenizer:
    def __init__(
        self,
        merge_operations: int,
        end_of_word: Optional[str] = "</w>",
        min_pair_freq: int = 2,
        pretokenizer: Optional[Callable[[str], Sequence[str]]] = None,   # [gen]
        symbolizer: Optional[Callable[[str], Sequence[str]]] = None,     # [gen]
    ):
        self.merge_operations = merge_operations
        self.end_of_word = end_of_word
        self.min_pair_freq = min_pair_freq
        self.merges = []                    # learned merge rules, in order
        self.merge_ranks = {}               # merge pair -> rank
        self.encoder_cache = {}             # segment -> bpe symbols
        # [gen] defaults reproduce the original behaviour (whitespace words, char symbols)
        self.pretokenizer = pretokenizer or (lambda line: line.split())
        self.symbolizer = symbolizer or (lambda seg: list(seg))

    def _init_symbols(self, seg: str) -> list:            # [gen] shared by train + encode
        syms = list(self.symbolizer(seg))
        if self.end_of_word is not None:
            syms.append(self.end_of_word)
        return syms

    def train(self, lines):
        """Incremental BPE: per merge, only the segments containing the winning
        pair are rewritten, and pair counts are patched rather than recomputed."""
        self.merges = []

        # 1) count segment types (training data only)
        seg_freq = Counter()
        for line in lines:
            for seg in self.pretokenizer(line):          # [gen]
                if seg:
                    seg_freq[seg] += 1

        # 2) segments as symbol lists + their frequencies
        seg_symbols, seg_freqs = [], []
        for seg, freq in seg_freq.items():
            seg_symbols.append(self._init_symbols(seg))  # [gen]
            seg_freqs.append(freq)

        # 3) pair counts + inverted index  pair -> {segment_id}
        pair_counts = Counter()
        pair_to_segs = defaultdict(set)
        for sid, symbols in enumerate(seg_symbols):
            freq = seg_freqs[sid]
            for p in self._pairs_in(symbols):
                pair_counts[p] += freq
                pair_to_segs[p].add(sid)

        # 4) merge loop
        for _ in tqdm(range(self.merge_operations), desc="bpe merges"):
            best_pair, best_count = self._best_pair(pair_counts)
            if best_pair is None or best_count < self.min_pair_freq:
                break

            affected = pair_to_segs.get(best_pair)
            if not affected:
                pair_counts.pop(best_pair, None)
                continue
            affected = list(affected)

            for sid in affected:
                old = seg_symbols[sid]
                freq = seg_freqs[sid]
                for p in self._pairs_in(old):            # remove old contributions
                    pair_counts[p] -= freq
                    pair_to_segs[p].discard(sid)
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                new = self._merge_symbols_once(old, best_pair)
                seg_symbols[sid] = new
                for p in self._pairs_in(new):            # add new contributions
                    pair_counts[p] += freq
                    pair_to_segs[p].add(sid)

            pair_counts.pop(best_pair, None)
            pair_to_segs.pop(best_pair, None)
            self.merge_ranks[best_pair] = len(self.merges)
            self.merges.append(best_pair)

    def encode_word(self, seg: str) -> list:
        """Greedy encode one segment by applying learned merges in rank order.
        NOTE: the end-of-word marker is kept in the output (the original stripped
        it, which loses word boundaries -- fatal for a seq2seq target)."""
        cached = self.encoder_cache.get(seg)
        if cached is not None:
            return cached

        symbols = self._init_symbols(seg)                # [gen]
        while True:
            best_rank = self.merge_operations + 1
            best_pair = None
            for p in self._pairs_in(symbols):
                if p in self.merge_ranks and self.merge_ranks[p] < best_rank:
                    best_rank = self.merge_ranks[p]
                    best_pair = p
            if best_pair is None:
                break
            symbols = self._merge_symbols_once(symbols, best_pair)

        self.encoder_cache[seg] = symbols                # [gen] no strip
        return symbols

    def encode_line(self, line: str) -> list:
        out = []
        for seg in self.pretokenizer(line):             # [gen]
            out.extend(self.encode_word(seg))
        return out

    @staticmethod
    def _pairs_in(symbols):
        for i in range(len(symbols) - 1):
            yield (symbols[i], symbols[i + 1])

    @staticmethod
    def _has_punct_or_symbol(sym) -> bool:
        for ch in sym:
            if unicodedata.category(ch)[0] in {"P", "S"}:
                return True
        return False

    def _best_pair(self, pair_counts):
        if not pair_counts:
            return None, 0
        best_pair, best_count = None, -1
        for (a, b), count in pair_counts.items():
            if a == self.end_of_word or b == self.end_of_word:
                continue
            if self._has_punct_or_symbol(a) or self._has_punct_or_symbol(b):
                continue
            if count > best_count:
                best_count = count
                best_pair = (a, b)
        return best_pair, best_count

    @staticmethod
    def _merge_symbols_once(symbols, pair):
        a, b = pair
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged


# =============================================================================
# pre-tokenizers / symbolizers
# =============================================================================
def plain_pretokenize(line: str) -> List[str]:
    return line.split()


def plain_symbolize(seg: str) -> List[str]:
    return list(seg)


def cipher_pretokenize(line: str) -> List[str]:
    return [line.strip()]                                # whole line = one segment


def cipher_symbolize(seg: str) -> List[str]:
    assert len(seg) % _BYTE == 0, f"cipher segment length {len(seg)} not a multiple of 8"
    return [seg[i:i + _BYTE] for i in range(0, len(seg), _BYTE)]


# =============================================================================
# Tokenizer wrapper: adds id<->symbol maps, specials, decode, save/load
# =============================================================================
class Tokenizer:
    def __init__(self, bpe: BPETokenizer, itos: List[str], kind: str):
        self.bpe = bpe
        self.kind = kind                                 # "plain" | "cipher"
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    # ---- construction ----
    @classmethod
    def build(cls, kind: str, train_lines: Sequence[str], merge_ops: int) -> "Tokenizer":
        if kind == "plain":
            bpe = BPETokenizer(merge_ops, end_of_word="</w>",
                               pretokenizer=plain_pretokenize, symbolizer=plain_symbolize)
        elif kind == "cipher":
            bpe = BPETokenizer(merge_ops, end_of_word=None,
                               pretokenizer=cipher_pretokenize, symbolizer=cipher_symbolize)
        else:
            raise ValueError(kind)

        bpe.train(list(train_lines))

        # vocab = specials + every symbol the trained BPE emits on the corpus,
        # ordered by descending frequency (ties broken lexically for determinism)
        counts = Counter()
        for line in train_lines:
            for sym in bpe.encode_line(line):
                counts[sym] += 1
        if kind == "cipher":                             # guarantee all 256 bytes exist
            for b in range(256):
                counts.setdefault(format(b, "08b"), 0)
        symbols = sorted(counts, key=lambda s: (-counts[s], s))
        itos = list(SPECIALS) + symbols
        return cls(bpe, itos, kind)

    # ---- encode / decode ----
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(s, UNK_ID) for s in self.bpe.encode_line(text)]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], strip: bool = True) -> str:
        specials = {PAD_ID, BOS_ID, EOS_ID}
        syms = [self.itos[i] for i in ids if 0 <= i < len(self.itos) and i not in specials]
        text = "".join("" if s == "<unk>" else s for s in syms)
        if self.kind == "plain":
            text = text.replace("</w>", " ")
            if strip:  # keep boundary spaces when reassembling chunks
                text = text.strip()
        return text

    # ---- persistence ----
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "kind": self.kind,
                    "itos": self.itos,
                    "merge_operations": self.bpe.merge_operations,
                    "end_of_word": self.bpe.end_of_word,
                    "merges": self.bpe.merges,
                    "merge_ranks": {"\x00".join(k): v for k, v in self.bpe.merge_ranks.items()},
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        kind = d["kind"]
        pre, sym = (
            (plain_pretokenize, plain_symbolize) if kind == "plain"
            else (cipher_pretokenize, cipher_symbolize)
        )
        bpe = BPETokenizer(d["merge_operations"], end_of_word=d["end_of_word"],
                           pretokenizer=pre, symbolizer=sym)
        bpe.merges = [tuple(m) for m in d["merges"]]
        bpe.merge_ranks = {tuple(k.split("\x00")): v for k, v in d["merge_ranks"].items()}
        return cls(bpe, d["itos"], kind)


class ByteCodec:
    """Token-free codec for C5/BLT. Same interface as Tokenizer (vocab_size,
    encode, decode) so datasets and generate_predictions treat it identically.
    ids: 0..3 = <pad>/<bos>/<eos>/<unk>; byte value v -> id v+4 (vocab 260)."""

    OFF = len(SPECIALS)  # 4

    def __init__(self, kind: str):
        assert kind in ("cipher", "plain")
        self.kind = kind
        self.vocab_size = 256 + self.OFF

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        if self.kind == "cipher":                       # text is a "0"/"1" string
            assert len(text) % 8 == 0
            vals = [int(text[i:i + 8], 2) for i in range(0, len(text), 8)]
        else:                                           # plaintext (ASCII: A-Za-z space)
            vals = [ord(c) & 0xFF for c in text]
        ids = [v + self.OFF for v in vals]
        return [BOS_ID] + ids + [EOS_ID] if add_bos_eos else ids

    def decode(self, ids: Sequence[int], strip: bool = True) -> str:
        vals = [i - self.OFF for i in ids if i >= self.OFF]
        if self.kind == "cipher":
            return "".join(f"{v:08b}" for v in vals)
        text = "".join(chr(v) for v in vals if 0 <= v < 256)
        return text.strip() if strip else text


# =============================================================================
# encoded dataset + collation
# =============================================================================
class Seq2SeqDataset(Dataset):
    """Holds encoded id lists plus the raw target strings (used as clean
    references at eval time, avoiding tokenizer round-trip artifacts)."""

    def __init__(
        self,
        src_ids: List[List[int]],
        tgt_ids: List[List[int]],
        src_text: List[str],
        tgt_text: List[str],
        group_id: Optional[List[int]] = None,   # chunk -> index of its source line
        line_text: Optional[List[str]] = None,  # full plaintext per group_id (for reassembly)
    ):
        self.src_ids, self.tgt_ids = src_ids, tgt_ids
        self.src_text, self.tgt_text = src_text, tgt_text
        self.group_id = group_id if group_id is not None else list(range(len(src_ids)))
        self.line_text = line_text if line_text is not None else list(tgt_text)

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, i: int):
        return (
            torch.tensor(self.src_ids[i], dtype=torch.long),
            torch.tensor(self.tgt_ids[i], dtype=torch.long),
        )


class TokenBudgetSampler:
    """Yields batches (lists of dataset indices) whose padded size --
    n_rows * longest_seq_in_batch -- stays under `max_tokens`. Examples are
    sorted by length so each batch is near-uniform (little wasted padding) and
    a batch holding a long example simply gets fewer rows. This is what keeps
    the O(batch * seq^2) attention from OOMing on the long-sequence tail.

    Batch *order* is reshuffled every epoch (call set_epoch); the within-batch
    grouping is fixed."""

    def __init__(self, lengths: Sequence[int], max_tokens: int,
                 shuffle: bool = True, seed: int = 0):
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self._batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0
        for i in order:
            nmax = max(cur_max, lengths[i])
            if cur and nmax * (len(cur) + 1) > max_tokens:
                self._batches.append(cur)
                cur, cur_max, nmax = [], 0, lengths[i]
            cur.append(i)
            cur_max = nmax
        if cur:
            self._batches.append(cur)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        yield from batches


def collate_seq2seq(batch, pad_id: int = PAD_ID):
    """Pad a batch to its own max source / target length."""
    srcs, tgts = zip(*batch)
    smax = max(s.size(0) for s in srcs)
    tmax = max(t.size(0) for t in tgts)
    src = torch.full((len(batch), smax), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), tmax), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : s.size(0)] = s
        tgt[i, : t.size(0)] = t
    return src, tgt                       # train.py slices tgt into (in, out)


# =============================================================================
# top-level pipeline
# =============================================================================
def read_pairs(cfg: Config) -> Tuple[List[str], List[str]]:
    cipher = [l.rstrip("\n") for l in open(os.path.join(cfg.data_dir, cfg.cipher_file))]
    plain = [l.rstrip("\n") for l in open(os.path.join(cfg.data_dir, cfg.plain_file))]
    assert len(cipher) == len(plain), "cipher/plain line count mismatch"
    return cipher, plain


def split_indices(n: int, n_val: int, n_test: int, seed: int):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return train, val, test


def _percentiles(xs: List[int], ps=(50, 90, 95, 99, 100)) -> str:
    xs = sorted(xs)
    return "  ".join(f"p{p}={xs[min(len(xs) - 1, int(len(xs) * p / 100))]}" for p in ps)


def build_or_load_tokenizers(
    cfg: Config, cipher_train: List[str], plain_train: List[str]
) -> Tuple[Tokenizer, Tokenizer]:
    os.makedirs(cfg.tokenizer_dir, exist_ok=True)
    src_path = os.path.join(cfg.tokenizer_dir, f"cipher_bpe_{cfg.src_merge_ops}.pkl")
    tgt_path = os.path.join(cfg.tokenizer_dir, f"plain_bpe_{cfg.tgt_merge_ops}.pkl")

    if os.path.exists(src_path):
        src_tok = Tokenizer.load(src_path)
    else:
        sub = cipher_train[:_CIPHER_BPE_TRAIN_LINES]
        print(f"[tok] training cipher BPE on {len(sub)} lines, {cfg.src_merge_ops} merges ...")
        src_tok = Tokenizer.build("cipher", sub, cfg.src_merge_ops)
        src_tok.save(src_path)

    if os.path.exists(tgt_path):
        tgt_tok = Tokenizer.load(tgt_path)
    else:
        print(f"[tok] training plain BPE on {len(plain_train)} lines, {cfg.tgt_merge_ops} merges ...")
        tgt_tok = Tokenizer.build("plain", plain_train, cfg.tgt_merge_ops)
        tgt_tok.save(tgt_path)

    print(f"[tok] cipher vocab={src_tok.vocab_size}  plain vocab={tgt_tok.vocab_size}")
    return src_tok, tgt_tok


def _encoded_cache_path(cfg: Config) -> str:
    key = "|".join(
        str(x)
        for x in (
            cfg.tokenization, cfg.cipher_file, cfg.plain_file,
            cfg.src_merge_ops, cfg.tgt_merge_ops,
            cfg.max_src_len, cfg.max_tgt_len, cfg.chunk_chars, cfg.seed,
            cfg.n_val, cfg.n_test, _CIPHER_BPE_TRAIN_LINES,
        )
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(cfg.tokenizer_dir, f"encoded_{h}.pt")


def _chunk_line(cipher: str, plain: str, w: int):
    """Split one aligned (cipher, plain) pair into windows of `w` plaintext chars
    (= w*8 cipher bits). `w % 8 == 0` guarantees every window starts at cipher
    phase 0. Returns [] -> [(cipher, plain)] when w <= 0."""
    if w <= 0:
        return [(cipher, plain)]
    return [(cipher[k * 8:(k + w) * 8], plain[k:k + w]) for k in range(0, len(plain), w)]


def make_datasets(cfg: Config):
    """Entry point for train.py. Returns (train_ds, val_ds, test_ds, src_tok, tgt_tok).

    Tokenizers are pickled per (merge_ops); the fully-encoded splits are cached
    to a .pt keyed on everything that affects them, so both the ~20 min BPE
    train and the corpus encode are paid once."""
    random.seed(cfg.seed)
    cipher, plain = read_pairs(cfg)
    tr, va, te = split_indices(len(cipher), cfg.n_val, cfg.n_test, cfg.seed)

    if cfg.tokenization == "blt":
        src_tok, tgt_tok = ByteCodec("cipher"), ByteCodec("plain")
        print(f"[tok] BLT byte codec  vocab={src_tok.vocab_size}")
    else:
        src_tok, tgt_tok = build_or_load_tokenizers(
            cfg, [cipher[i] for i in tr], [plain[i] for i in tr]
        )

    cache_path = _encoded_cache_path(cfg)
    _FIELDS = ("src_ids", "tgt_ids", "src_text", "tgt_text", "group_id", "line_text")
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        print(f"[data] loaded encoded splits from {cache_path}")
        ds = {k: Seq2SeqDataset(**v) for k, v in blob.items()}
        return ds["train"], ds["val"], ds["test"], src_tok, tgt_tok

    def encode_split(indices, split_name: str, drop_over_cap: bool):
        s_ids, t_ids, s_txt, t_txt, gid, ltext = [], [], [], [], [], []
        s_len, t_len = [], []
        dropped = 0
        for i in indices:
            g = len(ltext)                         # group index for this source line
            ltext.append(plain[i])
            for cc, pc in _chunk_line(cipher[i], plain[i], cfg.chunk_chars):
                si, ti = src_tok.encode(cc), tgt_tok.encode(pc)
                s_len.append(len(si)); t_len.append(len(ti))
                if len(si) > cfg.max_src_len or len(ti) > cfg.max_tgt_len:
                    if drop_over_cap:
                        dropped += 1
                        continue
                    si, ti = si[: cfg.max_src_len], ti[: cfg.max_tgt_len]
                s_ids.append(si); t_ids.append(ti)
                s_txt.append(cc); t_txt.append(pc); gid.append(g)
        print(
            f"[data] {split_name:5s} lines={len(indices):4d} chunks={len(s_ids):5d}  "
            f"src_tok[{_percentiles(s_len)}]  tgt_tok[{_percentiles(t_len)}]  "
            f"dropped_over_cap={dropped}"
        )
        return Seq2SeqDataset(s_ids, t_ids, s_txt, t_txt, gid, ltext)

    train_ds = encode_split(tr, "train", drop_over_cap=True)
    val_ds = encode_split(va, "val", drop_over_cap=True)
    test_ds = encode_split(te, "test", drop_over_cap=False)   # keep every test chunk

    torch.save(
        {name: {f: getattr(d, f) for f in _FIELDS}
         for name, d in (("train", train_ds), ("val", val_ds), ("test", test_ds))},
        cache_path,
    )
    print(f"[data] cached encoded splits -> {cache_path}")
    return train_ds, val_ds, test_ds, src_tok, tgt_tok
