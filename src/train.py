"""
train.py -- config-driven training loop for C1-C4 (subword seq2seq).

    python -m src.train --config C1
    python -m src.train --config C3 --device cuda --no-wandb

One run == one config. Teacher-forced cross-entropy with label smoothing,
AdamW, linear-warmup + inverse-sqrt (or cosine) LR. Periodic greedy-decode
eval on a val subset drives checkpointing and early stopping. Peak GPU memory
and step time are logged for the C5/BLT comparison later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import get_config
from .dataset import TokenBudgetSampler, collate_seq2seq, make_datasets
from .models.blt import ByteLatentTransformer
from .models.transformer import Seq2SeqTransformer
from .utils import (
    build_scheduler,
    compute_all_metrics,
    count_params,
    generate_predictions,
    peak_gpu_mem_mb,
    pick_device,
    plot_training_curves,
    reset_peak_gpu_mem,
    set_seed,
)

try:
    import wandb
except ImportError:
    wandb = None


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    # no weight decay on biases / norm gains / embeddings
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=cfg.adam_betas, eps=cfg.adam_eps,
    )


def compute_loss(model, src, tgt, criterion, is_blt: bool):
    """C1-C4: token-level teacher forcing. C5/BLT: model returns (byte_logits,
    labels) already patch-aligned, so no shift here."""
    if is_blt:
        logits, labels = model(src, tgt)
    else:
        logits, labels = model(src, tgt[:, :-1]), tgt[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


@torch.no_grad()
def val_loss(model, loader, criterion, device, is_blt: bool) -> float:
    model.eval()
    tot, n = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tot += compute_loss(model, src, tgt, criterion, is_blt).item() * src.size(0)
        n += src.size(0)
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["C1", "C2", "C3", "C4", "C5"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config(args.config)
    if args.max_steps:
        cfg = replace(cfg, max_steps=args.max_steps)
    if args.batch_size:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)

    device = pick_device(args.device)
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # --- data + tokenizers (vocab sizes only known now) ---
    train_ds, val_ds, test_ds, src_tok, tgt_tok = make_datasets(cfg)
    cfg = replace(cfg, src_vocab_size=src_tok.vocab_size, tgt_vocab_size=tgt_tok.vocab_size)

    def _lens(ds):
        return [max(len(s), len(t)) for s, t in zip(ds.src_ids, ds.tgt_ids)]

    train_sampler = TokenBudgetSampler(_lens(train_ds), cfg.max_tokens,
                                       shuffle=True, seed=cfg.seed)
    val_sampler = TokenBudgetSampler(_lens(val_ds), cfg.max_tokens, shuffle=False)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate_seq2seq, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                            collate_fn=collate_seq2seq, num_workers=cfg.num_workers)

    # --- model / optim ---
    is_blt = cfg.tokenization == "blt"
    model = (ByteLatentTransformer(cfg) if is_blt else Seq2SeqTransformer(cfg)).to(device)
    total, _ = count_params(model)
    print(f"[{cfg.name}] device={device}  params={total/1e6:.2f}M  "
          f"vocab src={cfg.src_vocab_size} tgt={cfg.tgt_vocab_size}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_id,
                                    label_smoothing=cfg.label_smoothing)
    # bf16 only where the GPU supports it natively (Ampere+); T4/Turing -> fp32
    use_amp = cfg.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()

    run = None
    if wandb is not None and not args.no_wandb:
        run = wandb.init(project=cfg.wandb_project, name=cfg.name, config=asdict(cfg))

    # --- loop ---
    history = {"train_loss": [], "val_loss": [], "val_seq_acc": []}
    # "best" is lexicographic (seq_acc, then -val_loss): while seq_acc is pinned
    # at 0 early on, a falling val_loss still advances the saved checkpoint, so
    # the final reload is never a stale early-step model.
    best_score, best_seq_acc, patience, step = (-1.0, -1e9), 0.0, 0, 0
    step_ema = None
    reset_peak_gpu_mem(device)
    stop = False
    epoch = 0

    while step < cfg.max_steps and not stop:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for src, tgt in train_loader:
            model.train()
            src, tgt = src.to(device), tgt.to(device)
            t0 = time.time()

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = compute_loss(model, src, tgt, criterion, is_blt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            step_ema = dt if step_ema is None else 0.9 * step_ema + 0.1 * dt

            if step % 20 == 0:
                n_tok = (tgt[:, 1:] != cfg.pad_id).sum().item()
                log = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(gnorm),
                    "perf/step_ms": step_ema * 1e3,
                    "perf/tok_per_s": n_tok / max(dt, 1e-6),
                    "perf/peak_mem_mb": peak_gpu_mem_mb(device),
                    "step": step,
                }
                history["train_loss"].append((step, loss.item()))
                if run:
                    run.log(log)
                print(f"step {step:5d} | loss {loss.item():.4f} | lr {log['train/lr']:.2e} "
                      f"| {log['perf/step_ms']:.0f} ms/step")

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                vloss = val_loss(model, val_loader, criterion, device, is_blt)
                preds, refs = generate_predictions(model, val_ds, tgt_tok, cfg, device,
                                                   subset=cfg.eval_subset,
                                                   max_len=cfg.eval_gen_max_len)
                vm = compute_all_metrics(preds, refs, include_bleu_rouge=True)
                history["val_loss"].append((step, vloss))
                history["val_seq_acc"].append((step, vm["seq_acc"]))
                print(f"  [eval @ {step}] val_loss {vloss:.4f} | "
                      + " ".join(f"{k}={v:.2f}" for k, v in vm.items()))
                if run:
                    run.log({"val/loss": vloss, **{f"val/{k}": v for k, v in vm.items()},
                             "step": step})

                score = (round(vm["seq_acc"], 2), -round(vloss, 4))
                if score > best_score:
                    best_score, best_seq_acc, patience = score, vm["seq_acc"], 0
                    torch.save({"model": model.state_dict(), "cfg": asdict(cfg),
                                "step": step, "val_metrics": vm},
                               os.path.join(cfg.ckpt_dir, f"{cfg.name}.pt"))
                else:
                    patience += 1
                    if step >= cfg.min_steps and patience >= cfg.early_stop_patience:
                        print(f"  early stop: no val gain (seq_acc/loss) in {patience} evals")
                        stop = True

            if step >= cfg.max_steps or stop:
                break

    # --- final: test-set eval on the best checkpoint ---
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.name}.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    preds, refs = generate_predictions(model, test_ds, tgt_tok, cfg, device, reassemble=True)
    test_m = compute_all_metrics(preds, refs, include_bleu_rouge=True)
    test_m["peak_mem_mb"] = peak_gpu_mem_mb(device)
    test_m["step_time_ms"] = (step_ema or 0.0) * 1e3
    print(f"[{cfg.name}] TEST: " + " ".join(f"{k}={v:.3f}" for k, v in test_m.items()))

    with open(os.path.join(cfg.out_dir, f"{cfg.name}_metrics.json"), "w") as f:
        json.dump({"config": cfg.name, "test": test_m,
                   "best_val_seq_acc": best_seq_acc, "steps": step}, f, indent=2)
    with open(os.path.join(cfg.out_dir, f"{cfg.name}_samples.txt"), "w") as f:
        for p, r in list(zip(preds, refs))[:20]:
            f.write(f"REF : {r}\nPRED: {p}\n\n")
    plot_training_curves(history, os.path.join(cfg.out_dir, f"{cfg.name}_curves.png"))
    if run:
        run.log({f"test/{k}": v for k, v in test_m.items()})
        run.finish()


if __name__ == "__main__":
    main()
