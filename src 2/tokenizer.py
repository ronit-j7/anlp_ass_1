"""Byte-Pair Encoding written from scratch: training, encoding and decoding.

No tokenizer library is used. Pre-tokenization only bounds where merges may happen
(whitespace for English, fixed windows for the binary string, which has no natural
boundaries); the subword units themselves are learned and variable length.
"""
import heapq
import json
import re
from collections import Counter, defaultdict

SPACE = "▁"                      # visible space marker, so decoding is exact
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
BIT_WINDOW = 32


def pretokenize_text(text: str):
    return [w.replace(" ", SPACE) for w in re.findall(r"\s*\S+|\s+", text)]


def pretokenize_bits(bits: str, window: int = BIT_WINDOW):
    return [bits[i:i + window] for i in range(0, len(bits), window)]


def _pairs(syms):
    return zip(syms, syms[1:])


class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab                                     # token -> id
        self.itos = {i: t for t, i in vocab.items()}
        self.ranks = {(a, b): i for i, (a, b) in enumerate(merges)}
        self.merges = merges
        self.pad, self.bos, self.eos, self.unk = (vocab[t] for t in SPECIALS)
        self._cache = {}

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, words: Counter, vocab_size: int, verbose: bool = True):
        """words: pre-token -> frequency. Merges the most frequent adjacent pair repeatedly."""
        splits = {w: tuple(w) for w in words}
        freq, where = Counter(), defaultdict(set)
        for w, f in words.items():
            for p in _pairs(splits[w]):
                freq[p] += f
                where[p].add(w)

        alphabet = sorted({c for w in words for c in w})
        vocab = {t: i for i, t in enumerate(SPECIALS + alphabet)}
        merges = []
        heap = [(-c, p) for p, c in freq.items()]
        heapq.heapify(heap)

        while len(vocab) < vocab_size and heap:
            negc, best = heapq.heappop(heap)
            if freq.get(best, 0) != -negc:                     # stale heap entry
                continue
            if -negc < 2:
                break
            new_tok = best[0] + best[1]
            vocab[new_tok] = len(vocab)
            merges.append(best)
            touched = set()
            for w in list(where[best]):
                old, f = splits[w], words[w]
                new, i = [], 0
                while i < len(old):
                    if i < len(old) - 1 and (old[i], old[i + 1]) == best:
                        new.append(new_tok); i += 2
                    else:
                        new.append(old[i]); i += 1
                new = tuple(new)
                if new == old:
                    continue
                for p in _pairs(old):
                    freq[p] -= f; touched.add(p)
                for p in _pairs(new):
                    freq[p] += f; where[p].add(w); touched.add(p)
                splits[w] = new
            freq.pop(best, None); where.pop(best, None)
            for p in touched:
                if freq.get(p, 0) > 0:
                    heapq.heappush(heap, (-freq[p], p))
            if verbose and len(vocab) % 1000 == 0:
                print(f"  bpe vocab {len(vocab)}/{vocab_size}", flush=True)
        return cls(vocab, merges)

    # ------------------------------------------------------------------ inference
    def _bpe(self, word: str):
        if word in self._cache:
            return self._cache[word]
        syms = list(word)
        while len(syms) > 1:
            ranked = [(self.ranks.get(p, 1 << 30), i) for i, p in enumerate(_pairs(syms))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            syms[i:i + 2] = [syms[i] + syms[i + 1]]
        self._cache[word] = syms
        return syms

    def encode(self, pieces):
        return [self.vocab.get(s, self.unk) for w in pieces for s in self._bpe(w)]

    def encode_text(self, text):
        return self.encode(pretokenize_text(text))

    def encode_bits(self, bits):
        return self.encode(pretokenize_bits(bits))

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids
                       if i not in (self.pad, self.bos, self.eos)).replace(SPACE, " ")

    def __len__(self):
        return len(self.vocab)

    # ------------------------------------------------------------------ io
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])
