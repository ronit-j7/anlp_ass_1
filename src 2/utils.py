"""Metrics (all implemented here, no external metric packages) and plotting helpers."""
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------ sequence-level metrics
def _bits(s: str) -> str:
    return "".join(format(b, "08b") for b in s.encode("utf-8"))


def bit_accuracy(preds: List[str], refs: List[str]) -> float:
    """Fraction of matching bits between the UTF-8 bit strings of prediction and reference.
    Length mismatches count as errors (denominator = longer of the two)."""
    match = total = 0
    for p, r in zip(preds, refs):
        pb, rb = _bits(p), _bits(r)
        match += sum(a == b for a, b in zip(pb, rb))
        total += max(len(pb), len(rb))
    return match / max(total, 1)


def sequence_accuracy(preds: List[str], refs: List[str]) -> float:
    return sum(p == r for p, r in zip(preds, refs)) / max(len(refs), 1)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def levenshtein_stats(preds: List[str], refs: List[str]):
    dists = [levenshtein(p, r) for p, r in zip(preds, refs)]
    norm = [d / max(len(r), 1) for d, r in zip(dists, refs)]
    return float(np.mean(dists)), float(np.mean(norm))


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(preds: List[str], refs: List[str], max_n: int = 4) -> float:
    """Standard corpus BLEU-4 (whitespace tokens, single reference, +1 smoothing on n>1)."""
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len = ref_len = 0
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        pred_len += len(pt)
        ref_len += len(rt)
        for n in range(1, max_n + 1):
            pn, rn = _ngrams(pt, n), _ngrams(rt, n)
            clipped[n - 1] += sum(min(c, rn[g]) for g, c in pn.items())
            totals[n - 1] += max(sum(pn.values()), 0)
    logs = []
    for n in range(max_n):
        num, den = clipped[n], totals[n]
        if den == 0:
            return 0.0
        if num == 0:                       # smoothing keeps short segments from zeroing out
            num, den = 1, den + 1
        logs.append(math.log(num / den))
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(sum(logs) / max_n)


def _lcs(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], (prev + 1 if x == y else max(dp[j], dp[j - 1]))
    return dp[-1]


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0:
        return 0.0
    prec, rec = overlap / n_pred, overlap / n_ref
    return 2 * prec * rec / (prec + rec)


def rouge_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """ROUGE-1/2/L F1, averaged over examples."""
    r1 = r2 = rl = 0.0
    for p, r in zip(preds, refs):
        pt, rt = p.split(), r.split()
        if not pt or not rt:
            continue
        for n, acc in ((1, "r1"), (2, "r2")):
            pn, rn = _ngrams(pt, n), _ngrams(rt, n)
            ov = sum(min(c, rn[g]) for g, c in pn.items())
            f = _f1(ov, max(sum(pn.values()), 1), max(sum(rn.values()), 1))
            if acc == "r1":
                r1 += f
            else:
                r2 += f
        rl += _f1(_lcs(pt, rt), len(pt), len(rt))
    n = max(len(refs), 1)
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    lev, lev_norm = levenshtein_stats(preds, refs)
    m = {
        "bit_accuracy": bit_accuracy(preds, refs),
        "sequence_accuracy": sequence_accuracy(preds, refs),
        "levenshtein": lev,
        "levenshtein_norm": lev_norm,
        "bleu": corpus_bleu(preds, refs),
    }
    m.update(rouge_scores(preds, refs))
    return m


# ------------------------------------------------------------------ plots
def plot_all(outputs_dir: str = "outputs"):
    """Build comparison plots from every outputs/metrics_C*.json produced by train.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {}
    for name in sorted(os.listdir(outputs_dir)):
        if name.startswith("metrics_") and name.endswith(".json"):
            with open(os.path.join(outputs_dir, name)) as f:
                r = json.load(f)
            runs[r["config"]] = r
    if not runs:
        print("no metrics files found")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cfg, r in runs.items():
        ep = range(1, len(r["history"]["train_loss"]) + 1)
        ax[0].plot(ep, r["history"]["train_loss"], label=cfg)
        ax[1].plot(ep, r["history"]["val_loss"], label=cfg)
    for a, t in zip(ax, ["Training loss", "Validation loss"]):
        a.set_xlabel("epoch"); a.set_ylabel("cross-entropy"); a.set_title(t); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outputs_dir, "loss_curves.png"), dpi=150)

    keys = ["bit_accuracy", "sequence_accuracy", "levenshtein_norm", "bleu", "rougeL"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.4))
    cfgs = sorted(runs)
    for a, k in zip(axes, keys):
        a.bar(cfgs, [runs[c]["test"][k] for c in cfgs], color="#4C72B0")
        a.set_title(k); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outputs_dir, "metric_comparison.png"), dpi=150)

    # controlled benchmark if available, else the (shared-GPU) wall-clock numbers
    bench_path = os.path.join(outputs_dir, "benchmark.json")
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bench = {r["config"]: r for r in json.load(f)}
        cost_keys = [("ms_per_step", "Training ms / step"), ("train_peak_mem_mb", "Peak GPU memory (MB)"),
                     ("decode_ms_per_example", "Greedy decode ms / example"), ("params", "Parameters")]
        src = bench
    else:
        cost_keys = [("train_time_s", "Training time (s)"), ("peak_mem_mb", "Peak GPU memory (MB)"),
                     ("params", "Parameters")]
        src = runs
    fig, axes = plt.subplots(1, len(cost_keys), figsize=(4 * len(cost_keys), 3.4))
    for a, (k, t) in zip(axes, cost_keys):
        a.bar(cfgs, [src[c][k] for c in cfgs], color="#DD8452")
        a.set_title(t); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outputs_dir, "cost_comparison.png"), dpi=150)

    header = ["config", "params", "train_time_s", "peak_mem_mb"] + keys
    rows = [header] + [[c, runs[c]["params"], round(runs[c]["train_time_s"], 1),
                        round(runs[c]["peak_mem_mb"], 1)] + [round(runs[c]["test"][k], 4) for k in keys]
                       for c in cfgs]
    with open(os.path.join(outputs_dir, "results_table.md"), "w") as f:
        f.write("| " + " | ".join(map(str, rows[0])) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in rows[1:]:
            f.write("| " + " | ".join(map(str, row)) + " |\n")
    print(f"wrote plots and results_table.md to {outputs_dir}/")


if __name__ == "__main__":
    import sys
    plot_all(sys.argv[1] if len(sys.argv) > 1 else "outputs")
