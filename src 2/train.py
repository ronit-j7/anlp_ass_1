"""Train one of the five ablation configurations (C1-C5) on the cipher -> plaintext task."""
import argparse
import json
import math
import os
import sys
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset as D
from models.blt import BLTSeq2Seq
from models.transformer import ModelConfig, Seq2SeqTransformer
from utils import compute_metrics, count_params, set_seed

# Each configuration differs from the C1 base by exactly one component.
CONFIGS = {
    "C1": dict(pos="sinusoidal", attn="mha", norm="layernorm", tok="bpe"),
    "C2": dict(pos="rope",       attn="mha", norm="layernorm", tok="bpe"),
    "C3": dict(pos="sinusoidal", attn="gqa", norm="layernorm", tok="bpe"),
    "C4": dict(pos="sinusoidal", attn="mha", norm="rmsnorm",   tok="bpe"),
    "C5": dict(pos="sinusoidal", attn="mha", norm="layernorm", tok="blt"),
}


def require_cuda(retries: int = 24, wait: int = 300):
    """These GPUs are shared and occasionally refuse a new CUDA context. Torch caches that
    failure, so wait and re-exec rather than silently training on the CPU."""
    if torch.cuda.is_available():
        return
    n = int(os.environ.get("CUDA_WAIT_TRIES", "0"))
    if n >= retries:
        raise RuntimeError("no usable GPU after waiting")
    print(f"no CUDA context yet; retry {n + 1}/{retries} in {wait}s", flush=True)
    time.sleep(wait)
    os.environ["CUDA_WAIT_TRIES"] = str(n + 1)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="C1", choices=list(CONFIGS))
    p.add_argument("--data_dir", default="Dataset_A1")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_kv_heads", type=int, default=2)      # used by GQA (C3)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target_patch", type=float, default=4.0)  # BLT: mean patch length
    p.add_argument("--n_local_layers", type=int, default=1)
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--max_src", type=int, default=192)
    p.add_argument("--max_tgt", type=int, default=48)
    p.add_argument("--max_eval", type=int, default=2000)
    p.add_argument("--max_train", type=int, default=0)     # >0 truncates for debugging
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="anlp-a1-transformers")
    p.add_argument("--wandb_mode", default="offline", choices=["online", "offline", "disabled"])
    return p.parse_args()


def build_data(args, cfg):
    pairs = D.read_pairs(args.data_dir)
    tr_l, va_l, te_l = D.split_by_line(pairs, seed=args.seed)
    tr, va, te = (D.chunk_pairs(x) for x in (tr_l, va_l, te_l))
    if args.max_train:
        tr, va, te = tr[:args.max_train], va[:args.max_train], te[:args.max_train]
    print(f"chunks  train {len(tr)}  val {len(va)}  test {len(te)}")

    if cfg["tok"] == "bpe":
        tok = D.build_tokenizer(tr, os.path.join(args.out_dir, "tokenizer.json"), args.vocab_size)
        make = lambda c: D.TokenizedDataset(c, tok, args.max_src, args.max_tgt)
        collate = partial(D.collate_tokenized, pad_id=tok.pad, bos_id=tok.bos, eos_id=tok.eos)
        info = dict(vocab=len(tok), pad=tok.pad, bos=tok.bos, eos=tok.eos, tokenizer=tok,
                    gen_len=args.max_tgt, patchers=None)
    else:
        patchers = D.build_patchers(tr, args.target_patch)
        make = lambda c: D.ByteDataset(c, patchers)
        collate = D.collate_bytes
        info = dict(vocab=D.BYTE_VOCAB, pad=D.BYTE_PAD, bos=D.BYTE_BOS, eos=D.BYTE_EOS,
                    tokenizer=None, gen_len=D.CHUNK_CHARS + 1, patchers=patchers)
    return [make(x) for x in (tr, va, te)], collate, info


def build_model(args, cfg, info):
    mc = ModelConfig(
        src_vocab=info["vocab"], tgt_vocab=info["vocab"], d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads if cfg["attn"] == "gqa" else args.n_heads,
        d_ff=args.d_ff, dropout=args.dropout, pos=cfg["pos"], norm=cfg["norm"],
        max_len=1024, n_local_layers=args.n_local_layers)
    if cfg["tok"] == "blt":
        (_, _), (tgt_model, tgt_theta) = info["patchers"]
        return BLTSeq2Seq(mc, info["vocab"], info["pad"], info["bos"], info["eos"],
                          patcher=tgt_model, theta=tgt_theta)
    return Seq2SeqTransformer(mc, info["pad"], info["bos"], info["eos"])


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    tot = n = 0.0
    for batch, _ in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tgt_out = batch["tgt_out"]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**{k: v for k, v in batch.items() if k != "tgt_out"})
        tot += loss_fn(logits.float().flatten(0, 1), tgt_out.flatten()).item() * tgt_out.numel()
        n += tgt_out.numel()
    return tot / max(n, 1)


def ids_to_text(ids, info, is_blt: bool):
    ids = list(ids)
    if info["eos"] in ids:
        ids = ids[:ids.index(info["eos"])]
    ids = [i for i in ids if i not in (info["pad"], info["bos"])]
    if is_blt:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="ignore")
    return info["tokenizer"].decode(ids)


@torch.no_grad()
def evaluate(model, dataset, collate, info, args, is_blt, device, limit):
    model.eval()
    subset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, collate_fn=collate)
    preds, refs = [], []
    for batch, texts in loader:
        inputs = [batch[k].to(device) for k in model.ENCODE_KEYS]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.greedy_decode(*inputs, info["gen_len"])
        preds += [ids_to_text(row, info, is_blt) for row in out.tolist()]
        refs += texts
    return compute_metrics(preds, refs), list(zip(preds[:10], refs[:10]))


def main():
    args = get_args()
    cfg = CONFIGS[args.config]
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    require_cuda()
    device = torch.device("cuda")
    is_blt = cfg["tok"] == "blt"

    (train_ds, val_ds, test_ds), collate, info = build_data(args, cfg)
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=sh, collate_fn=collate,
                          num_workers=2, pin_memory=True, drop_last=sh)
               for ds, sh in ((train_ds, True), (val_ds, False))]
    train_loader, val_loader = loaders

    model = build_model(args, cfg, info).to(device)
    n_params = count_params(model)
    print(f"[{args.config}] {cfg} | params {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98))
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, warmup=args.warmup, total=total_steps))
    loss_fn = nn.CrossEntropyLoss(ignore_index=info["pad"])

    import wandb
    wandb.init(project=args.wandb_project, name=args.config, mode=args.wandb_mode,
               dir=args.out_dir, config={**vars(args), **cfg, "params": n_params})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history = {"train_loss": [], "val_loss": []}
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = n_tok = 0.0
        for batch, _ in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            tgt_out = batch["tgt_out"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**{k: v for k, v in batch.items() if k != "tgt_out"})
            loss = loss_fn(logits.float().flatten(0, 1), tgt_out.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step(); step += 1
            run_loss += loss.item() * tgt_out.numel(); n_tok += tgt_out.numel()
            if step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "lr": sched.get_last_lr()[0], "step": step})
        tr_loss = run_loss / n_tok
        va_loss = eval_loss(model, val_loader, loss_fn, device)
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        wandb.log({"epoch": epoch, "train/epoch_loss": tr_loss, "val/loss": va_loss,
                   "val/ppl": math.exp(min(va_loss, 20))})
        print(f"epoch {epoch:2d} | train {tr_loss:.4f} | val {va_loss:.4f} | {time.time()-t0:.0f}s")

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0

    test_metrics, samples = evaluate(model, test_ds, collate, info, args, is_blt, device, args.max_eval)
    val_metrics, _ = evaluate(model, val_ds, collate, info, args, is_blt, device, 512)
    print(f"[{args.config}] test {json.dumps(test_metrics, indent=None)}")
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.log({"train_time_s": train_time, "peak_mem_mb": peak_mem})

    if info["patchers"] is not None:
        cfg = {**cfg, "theta_src": info["patchers"][0][1], "theta_tgt": info["patchers"][1][1]}
    result = dict(config=args.config, settings=cfg, params=n_params, train_time_s=train_time,
                  peak_mem_mb=peak_mem, history=history, test=test_metrics, val=val_metrics,
                  samples=samples, args=vars(args))
    with open(os.path.join(args.out_dir, f"metrics_{args.config}.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"model": model.state_dict(), "config": cfg, "args": vars(args)},
               os.path.join(args.out_dir, f"ckpt_{args.config}.pt"))
    wandb.finish()


if __name__ == "__main__":
    main()
