"""
utils.py -- metrics, eval decoding, schedules, and plots.

Metrics operate on *decoded strings* (predicted vs reference plaintext):
    bit_level_accuracy   -- % matching bits of the UTF-8 byte encodings
    sequence_accuracy    -- % of exact full-string matches
    levenshtein          -- char edit distance (raw mean + length-normalized)
    corpus_bleu          -- BLEU-4, brevity penalty + add-eps smoothing
    corpus_rouge_l       -- LCS-based ROUGE-L F1 (whitespace tokens)
BLEU/ROUGE are reported for the tokenized models only (C1-C4), per the brief.
Hand-rolled and dependency-free; swap in sacrebleu / rouge-score if canonical
numbers are wanted.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import Config


# =============================================================================
# string / bit helpers
# =============================================================================
def _to_bits(b: bytes) -> str:
    return "".join(f"{x:08b}" for x in b)


def text_to_bits(s: str, encoding: str = "utf-8") -> str:
    return _to_bits(s.encode(encoding, errors="replace"))


def cut_at_eos(ids: Sequence[int], eos_id: int) -> List[int]:
    """Truncate a generated id sequence at the first EOS (exclusive)."""
    out = []
    for i in ids:
        if i == eos_id:
            break
        out.append(int(i))
    return out


# =============================================================================
# metrics
# =============================================================================
def bit_level_accuracy(preds: Sequence[str], refs: Sequence[str], encoding: str = "utf-8") -> float:
    """% of matching bits over the UTF-8 encodings. Sequences are compared up to
    the longer length; positions past the shorter one count as mismatches."""
    match = total = 0
    for p, r in zip(preds, refs):
        pb, rb = text_to_bits(p, encoding), text_to_bits(r, encoding)
        n = max(len(pb), len(rb))
        for k in range(n):
            if k < len(pb) and k < len(rb) and pb[k] == rb[k]:
                match += 1
        total += n
    return 100.0 * match / max(total, 1)


def sequence_accuracy(preds: Sequence[str], refs: Sequence[str]) -> float:
    if not preds:
        return 0.0
    return 100.0 * sum(p == r for p, r in zip(preds, refs)) / len(preds)


def levenshtein(a: str, b: str) -> int:
    """Char-level edit distance, two-row DP (O(min(len)) memory)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def corpus_levenshtein(preds: Sequence[str], refs: Sequence[str]) -> Tuple[float, float]:
    """Returns (mean raw distance, length-normalized distance ~ CER)."""
    dists = [levenshtein(p, r) for p, r in zip(preds, refs)]
    ref_chars = sum(len(r) for r in refs)
    mean_raw = sum(dists) / max(len(dists), 1)
    norm = sum(dists) / max(ref_chars, 1)
    return mean_raw, norm


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(preds: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    """Simplified corpus BLEU-4: clipped n-gram precision, geometric mean,
    brevity penalty. add-1e-9 smoothing so a single missing high-order n-gram
    does not zero the whole score."""
    p_num = [0] * max_n
    p_den = [0] * max_n
    pred_len = ref_len = 0
    for pred, ref in zip(preds, refs):
        pt, rt = pred.split(), ref.split()
        pred_len += len(pt)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            pc = _ngram_counts(pt, n)
            rc = _ngram_counts(rt, n)
            p_num[n - 1] += sum(min(c, rc[g]) for g, c in pc.items())
            p_den[n - 1] += max(sum(pc.values()), 0)

    precisions = [(num + 1e-9) / (den + 1e-9) for num, den in zip(p_num, p_den)]
    if min(precisions) <= 0:
        return 0.0
    gm = math.exp(sum(math.log(pr) for pr in precisions) / max_n)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return 100.0 * bp * gm


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            prev, dp[j] = dp[j], (prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1]))
    return dp[-1]


def corpus_rouge_l(preds: Sequence[str], refs: Sequence[str], beta: float = 1.2) -> float:
    """Mean sentence-level ROUGE-L F1 over whitespace tokens."""
    scores = []
    for pred, ref in zip(preds, refs):
        pt, rt = pred.split(), ref.split()
        if not pt or not rt:
            scores.append(0.0)
            continue
        l = _lcs_len(pt, rt)
        if l == 0:
            scores.append(0.0)
            continue
        prec, rec = l / len(pt), l / len(rt)
        scores.append((1 + beta**2) * prec * rec / (rec + beta**2 * prec))
    return 100.0 * sum(scores) / max(len(scores), 1)


def compute_all_metrics(
    preds: Sequence[str], refs: Sequence[str], include_bleu_rouge: bool = True
) -> Dict[str, float]:
    lev_raw, lev_norm = corpus_levenshtein(preds, refs)
    m = {
        "bit_acc": bit_level_accuracy(preds, refs),
        "seq_acc": sequence_accuracy(preds, refs),
        "lev_raw": lev_raw,
        "lev_norm": lev_norm,
    }
    if include_bleu_rouge:
        m["bleu"] = corpus_bleu(preds, refs)
        m["rouge_l"] = corpus_rouge_l(preds, refs)
    return m


# =============================================================================
# eval decoding (shared by train.py periodic eval and evaluate.py)
# =============================================================================
@torch.no_grad()
def generate_predictions(
    model,
    dataset,
    tgt_tok,
    cfg: Config,
    device,
    subset: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Greedy-decode `subset` (or all) examples of `dataset`; return (preds, refs)."""
    model.eval()
    n = len(dataset) if subset is None else min(subset, len(dataset))
    bs = batch_size or cfg.batch_size
    preds: List[str] = []
    for start in range(0, n, bs):
        rows = list(range(start, min(start + bs, n)))
        smax = max(len(dataset.src_ids[i]) for i in rows)
        src = torch.full((len(rows), smax), cfg.pad_id, dtype=torch.long)
        for b, i in enumerate(rows):
            ids = dataset.src_ids[i]
            src[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        gen = model.generate(src.to(device), max_len=cfg.gen_max_len)  # (B, L)
        for row in gen.tolist():
            body = cut_at_eos(row[1:], cfg.eos_id)  # drop BOS, stop at EOS
            preds.append(tgt_tok.decode(body))
    refs = [dataset.tgt_text[i] for i in range(n)]
    return preds, refs


# =============================================================================
# training helpers
# =============================================================================
def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_scheduler(optimizer, cfg: Config):
    """Linear warmup, then inverse-sqrt (Noam-style) or cosine decay to ~0."""
    warm = max(cfg.warmup_steps, 1)

    def inverse_sqrt(step: int) -> float:
        return step / warm if step < warm else (warm / max(step, 1)) ** 0.5

    def cosine(step: int) -> float:
        if step < warm:
            return step / warm
        prog = min((step - warm) / max(cfg.max_steps - warm, 1), 1.0)
        return 0.5 * (1 + math.cos(math.pi * prog))

    fn = cosine if cfg.lr_schedule == "cosine" else inverse_sqrt
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def reset_peak_gpu_mem(device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_mem_mb(device) -> float:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


# =============================================================================
# plots  (matplotlib imported lazily so the core code has no hard dep)
# =============================================================================
def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("matplotlib needed for plotting: pip install matplotlib") from e


def plot_training_curves(history: Dict[str, List[Tuple[int, float]]], out_path: str) -> None:
    """history: name -> list of (step, value). Train loss on the left axis,
    everything else on the right."""
    plt = _plt()
    fig, ax_l = plt.subplots(figsize=(7, 4.5))
    ax_r = ax_l.twinx()
    for name, series in history.items():
        if not series:
            continue
        xs, ys = zip(*series)
        ax = ax_l if name == "train_loss" else ax_r
        ax.plot(xs, ys, label=name, marker="" if name == "train_loss" else "o", ms=3)
    ax_l.set_xlabel("step")
    ax_l.set_ylabel("train loss")
    ax_r.set_ylabel("val metric")
    lines = ax_l.get_lines() + ax_r.get_lines()
    ax_l.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_ablation_bars(results: Dict[str, Dict[str, float]], metrics: Sequence[str], out_path: str) -> None:
    """results: config name -> {metric: value}. One grouped bar chart."""
    plt = _plt()
    names = list(results)
    ncol = len(metrics)
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.6), squeeze=False)
    for k, metric in enumerate(metrics):
        ax = axes[0][k]
        vals = [results[n].get(metric, 0.0) for n in names]
        ax.bar(names, vals, color="#4C78A8")
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_c5_tradeoff(results: Dict[str, Dict[str, float]], out_path: str) -> None:
    """Scatter seq_acc vs {peak memory, step time} to visualize the BLT tradeoff."""
    plt = _plt()
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, xkey, xlabel in ((axes[0], "peak_mem_mb", "peak GPU mem (MB)"),
                             (axes[1], "step_time_ms", "step time (ms)")):
        for n in names:
            ax.scatter(results[n].get(xkey, 0.0), results[n].get("seq_acc", 0.0))
            ax.annotate(n, (results[n].get(xkey, 0.0), results[n].get("seq_acc", 0.0)),
                        fontsize=8, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("sequence accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
