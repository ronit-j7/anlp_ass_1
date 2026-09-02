"""
evaluate.py -- standalone eval of a trained checkpoint, and cross-config rollup.

    python -m src.evaluate --config C1               # test-set metrics for one config
    python -m src.evaluate --config C2 --split val
    python -m src.evaluate --all                     # ablation table + plots across C1..C5

Greedy decoding only, per the brief. A checkpoint carries its own Config
(saved by train.py), so eval reconstructs the exact model without CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch

from .config import Config
from .dataset import make_datasets
from .models.blt import ByteLatentTransformer
from .models.transformer import Seq2SeqTransformer
from .utils import (
    compute_all_metrics,
    generate_predictions,
    peak_gpu_mem_mb,
    pick_device,
    plot_ablation_bars,
    plot_c5_tradeoff,
    reset_peak_gpu_mem,
    set_seed,
)

_ALL = ["C1", "C2", "C3", "C4", "C5"]
_TABLE_METRICS = ["bit_acc", "seq_acc", "lev_norm", "bleu", "rouge_l",
                  "peak_mem_mb", "step_time_ms"]


def evaluate_one(name: str, split: str, ckpt_path: str, device_pref: str) -> dict:
    device = pick_device(device_pref)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    set_seed(cfg.seed)

    train_ds, val_ds, test_ds, _src_tok, tgt_tok = make_datasets(cfg)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]

    is_blt = cfg.tokenization == "blt"
    model = (ByteLatentTransformer(cfg) if is_blt else Seq2SeqTransformer(cfg)).to(device)
    model.load_state_dict(ckpt["model"])

    reset_peak_gpu_mem(device)
    t0 = time.time()
    preds, refs = generate_predictions(model, ds, tgt_tok, cfg, device,
                                       reassemble=(getattr(cfg, "chunk_chars", 0) > 0))
    dt = time.time() - t0

    m = compute_all_metrics(preds, refs, include_bleu_rouge=(cfg.tokenization == "subword"))
    m["peak_mem_mb"] = peak_gpu_mem_mb(device)
    m["decode_s"] = dt
    m["n"] = len(preds)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"{name}_eval_{split}.json"), "w") as f:
        json.dump({"config": name, "split": split, "ckpt": ckpt_path,
                   "step": ckpt.get("step"), "metrics": m}, f, indent=2)
    with open(os.path.join(cfg.out_dir, f"{name}_eval_{split}_samples.txt"), "w") as f:
        for p, r in list(zip(preds, refs))[:30]:
            f.write(f"REF : {r}\nPRED: {p}\n\n")

    print(f"[{name}/{split}] " + "  ".join(f"{k}={m[k]:.3f}" for k in m if k != "n"))
    return m


def rollup(device_pref: str) -> None:
    """Collect per-config test metrics -- preferring the richer files train.py
    wrote, re-evaluating a checkpoint only if that file is missing -- then emit
    a CSV and the ablation plots."""
    results: dict = {}
    for name in _ALL:
        metrics_file = os.path.join("outputs", f"{name}_metrics.json")
        ckpt_file = os.path.join("checkpoints", f"{name}.pt")
        if os.path.exists(metrics_file):
            results[name] = json.load(open(metrics_file))["test"]
        elif os.path.exists(ckpt_file):
            results[name] = evaluate_one(name, "test", ckpt_file, device_pref)
        else:
            print(f"  skip {name}: no outputs/{name}_metrics.json or checkpoint")

    if not results:
        print("nothing to roll up")
        return

    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "ablation_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + _TABLE_METRICS)
        for name, m in results.items():
            w.writerow([name] + [f"{m.get(k, float('nan')):.4f}" for k in _TABLE_METRICS])
    print(f"\nwrote {csv_path}")
    hdr = f"{'cfg':<5}" + "".join(f"{k:>13}" for k in _TABLE_METRICS)
    print(hdr)
    for name, m in results.items():
        print(f"{name:<5}" + "".join(f"{m.get(k, float('nan')):>13.3f}" for k in _TABLE_METRICS))

    plot_ablation_bars(results, ["bit_acc", "seq_acc", "lev_norm", "bleu"],
                       os.path.join("outputs", "ablation_bars.png"))
    plot_c5_tradeoff(results, os.path.join("outputs", "c5_tradeoff.png"))
    print("wrote outputs/ablation_bars.png, outputs/c5_tradeoff.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=_ALL)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None, help="override checkpoints/<config>.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--all", action="store_true", help="cross-config rollup + plots")
    args = ap.parse_args()

    if args.all:
        rollup(args.device)
        return
    if not args.config:
        ap.error("give --config NAME or --all")
    ckpt = args.ckpt or os.path.join("checkpoints", f"{args.config}.pt")
    if not os.path.exists(ckpt):
        ap.error(f"checkpoint not found: {ckpt}")
    evaluate_one(args.config, args.split, ckpt, args.device)


if __name__ == "__main__":
    main()
