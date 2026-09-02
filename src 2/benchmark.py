"""Controlled speed / memory comparison of C1-C5.

The wall-clock times in metrics_*.json come from a shared GPU, so they are not comparable
across configurations. This runs every configuration back to back under identical conditions:
same batch size, same number of steps, one process at a time.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train as T
from utils import count_params, set_seed


def bench_one(name, args, device, steps, warmup):
    cfg = T.CONFIGS[name]
    set_seed(args.seed)
    (train_ds, _, test_ds), collate, info = T.build_data(args, cfg)
    model = T.build_model(args, cfg, info).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=info["pad"])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    seen = src_len = tgt_len = 0
    t0 = None
    for i, (batch, _) in enumerate(loader):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        if i == warmup + steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**{k: v for k, v in batch.items() if k != "tgt_out"})
        loss = loss_fn(logits.float().flatten(0, 1), batch["tgt_out"].flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if i >= warmup:
            seen += batch["src"].size(0)
            src_len += batch["src"].size(1); tgt_len += batch["tgt_out"].size(1)
    torch.cuda.synchronize()
    train_s = time.time() - t0
    train_mem = torch.cuda.max_memory_allocated() / 2 ** 20

    # greedy decoding cost (BLT emits bytes, the subword models emit tokens)
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    sub = torch.utils.data.Subset(test_ds, range(args.batch_size * 4))
    dec_loader = DataLoader(sub, batch_size=args.batch_size, collate_fn=collate)
    torch.cuda.synchronize(); t1 = time.time()
    n = 0
    for batch, _ in dec_loader:
        inputs = [batch[k].to(device) for k in model.ENCODE_KEYS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.greedy_decode(*inputs, info["gen_len"])
        n += inputs[0].size(0)
    torch.cuda.synchronize()
    dec_s = time.time() - t1

    return dict(config=name, params=count_params(model),
                steps_per_s=steps / train_s, examples_per_s=seen / train_s,
                ms_per_step=1000 * train_s / steps, train_peak_mem_mb=train_mem,
                decode_peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                decode_ms_per_example=1000 * dec_s / n,
                mean_src_len=src_len / steps, mean_tgt_len=tgt_len / steps)


def _fmt(v):
    return f"{v:,.1f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", default="../outputs/benchmark.json")
    extra, rest = p.parse_known_args()
    import sys
    sys.argv = [sys.argv[0]] + rest
    args = T.get_args()
    device = torch.device("cuda")

    rows = [bench_one(c, args, device, extra.steps, extra.warmup) for c in T.CONFIGS]
    for r in rows:
        print(f"{r['config']}  {r['ms_per_step']:7.1f} ms/step  {r['examples_per_s']:8.1f} ex/s  "
              f"{r['train_peak_mem_mb']:8.1f} MB  decode {r['decode_ms_per_example']:6.1f} ms/ex")
    with open(extra.out, "w") as f:
        json.dump(rows, f, indent=2)
    cols = [("config", "config"), ("params", "params"), ("ms_per_step", "ms/step"),
            ("examples_per_s", "examples/s"), ("train_peak_mem_mb", "train peak MB"),
            ("decode_ms_per_example", "decode ms/example"), ("mean_src_len", "mean src len"),
            ("mean_tgt_len", "mean tgt len")]
    with open(extra.out.replace(".json", "_table.md"), "w") as f:
        f.write("| " + " | ".join(c[1] for c in cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(
                _fmt(r[k]) for k, _ in cols) + " |\n")
    print("wrote", extra.out)


if __name__ == "__main__":
    main()
