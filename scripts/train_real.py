"""Train a ready-to-use SmallM checkpoint on the prepared token corpus.

CPU-friendly: cosine LR with warmup, gradient accumulation, grad clipping,
periodic checkpointing. Designed to be stopped and resumed.

Usage:
    PYTHONPATH=src python3 scripts/train_real.py \
        --config configs/smallm_30m.json \
        --data-dir data/processed \
        --out checkpoints/smallm_30m \
        --steps 4000 --batch-size 16 --seq-len 256 --lr 6e-4 --accum 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig
from smallm.data import TokenCorpus
from smallm.model import build_model
from smallm.train import make_optimizer


def lr_at(step: int, total: int, warmup: int, base: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return base * (min_ratio + (1 - min_ratio) * cos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-dir", default=None,
                    help="also save weights-only snapshots here (for model soup)")
    ap.add_argument("--snapshot-every", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = ModelConfig.load(args.config)
    tok_meta = os.path.join(args.data_dir, "meta.json")
    corpus = TokenCorpus(os.path.join(args.data_dir, "tokens.bin"), tok_meta)
    print(f"corpus tokens = {corpus.n/1e6:.1f}M", flush=True)

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params = {n_params/1e6:.1f}M", flush=True)

    optimizer = make_optimizer(model, lr=args.lr)

    start_step = 0
    ckpt_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optimizer"])
        start_step = sd["step"]
        print(f"resumed from step {start_step}", flush=True)

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for _ in range(args.accum):
            batch = corpus.sample_batch(args.batch_size, args.seq_len)
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = out["loss"] / args.accum
            loss.backward()
            batch_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += batch_loss

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            avg = running / args.log_every
            tok_done = (step + 1 - start_step) * args.batch_size * args.seq_len * args.accum
            print(
                f"step {step+1:6d}/{args.steps}  loss {avg:.4f}  lr {lr:.2e}  "
                f"tok/s {tok_done/dt:,.0f}  elapsed {dt/60:.1f}m",
                flush=True,
            )
            running = 0.0

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            sd = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "step": step + 1, "config": cfg.to_dict()}
            torch.save(sd, ckpt_path)
            print(f"saved checkpoint at step {step+1}", flush=True)

        if args.snapshot_dir and args.snapshot_every and (step + 1) % args.snapshot_every == 0:
            os.makedirs(args.snapshot_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                       os.path.join(args.snapshot_dir, f"snap_{args.seed}_{step+1}.pt"))

    # final export: weights only + tokenizer + config = ready-to-use artifact
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(args.out, "model.pt"))
    tc = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tc):
        import shutil
        shutil.copy(tc, os.path.join(args.out, "tokenizer.json"))
    model.eval()
    with torch.no_grad():
        vals = []
        for _ in range(10):
            b = corpus.sample_batch(args.batch_size, args.seq_len)
            vals.append(float(model(input_ids=b["input_ids"], labels=b["labels"])["loss"]))
    final_loss = sum(vals) / len(vals)
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump({"params": n_params, "steps": args.steps, "final_loss": final_loss}, f, indent=2)
    print("done. wrote model.pt and tokenizer.json", flush=True)


if __name__ == "__main__":
    main()
