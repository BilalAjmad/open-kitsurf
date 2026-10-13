"""Distill a trained SmallM teacher into the Nova architecture.

The student (Nova) is trained to match the teacher's next-token distribution
(Kullback-Leibler divergence) while also minimizing its own cross-entropy on the
real tokens. This is knowledge distillation: it repairs the small gap left by
the weight transplant and makes Nova reproduce the teacher's behaviour with the
new recurrent-depth, parallel-residual, QK-norm computation.

Usage:
    python3 scripts/distill_nova.py \
        --teacher checkpoints/smallm_11m/model.pt \
        --init    checkpoints/nova_ready/model.pt \
        --data-dir data/processed --out checkpoints/nova_ready \
        --steps 800 --batch-size 8 --seq-len 256 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

from smallm.config import ModelConfig
from smallm.model import build_model
from smallm.train.pretrain import make_optimizer
from smallm.data import TokenCorpus


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + __import__("math").cos(__import__("math").pi * prog))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--init", required=True, help="Nova checkpoint to start from")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--temp", type=float, default=2.0, help="distillation temperature")
    ap.add_argument("--alpha", type=float, default=0.5, help="KD weight vs hard CE")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--snapshot-dir", default="", help="save snap_<step>.pt here for soup")
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device = {device}", flush=True)

    tck = torch.load(args.teacher, map_location="cpu")
    tcfg = ModelConfig.from_dict(tck["config"])
    teacher = build_model(tcfg).to(device)
    teacher.load_state_dict(tck["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sck = torch.load(args.init, map_location="cpu")
    scfg = ModelConfig.from_dict(sck["config"])
    student = build_model(scfg).to(device)
    student.load_state_dict(sck["model"])

    meta = os.path.join(args.data_dir, "meta.json")
    corpus = TokenCorpus(os.path.join(args.data_dir, "tokens.bin"), meta)
    print(f"corpus tokens = {corpus.n/1e6:.1f}M", flush=True)
    print(f"teacher {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M -> "
          f"student {sum(p.numel() for p in student.parameters())/1e6:.1f}M",
          flush=True)

    opt = make_optimizer(student, lr=args.lr)
    start_step = 0
    last_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last_path):
        sd = torch.load(last_path, map_location="cpu")
        student.load_state_dict(sd["model"])
        opt.load_state_dict(sd["optimizer"])
        start_step = sd["step"]
        print(f"resumed distillation from step {start_step}", flush=True)

    student.train()
    t0 = time.time()
    running = 0.0
    T = args.temp
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            b = corpus.sample_batch(args.batch_size, args.seq_len)
            ids = b["input_ids"].to(device)
            labs = b["labels"].to(device)
            with torch.no_grad():
                tlog = teacher(input_ids=ids)["logits"]
            slog = student(input_ids=ids)["logits"]
            # distillation: match teacher distribution
            kd = F.kl_div(
                F.log_softmax(slog / T, dim=-1),
                F.softmax(tlog / T, dim=-1),
                reduction="none",
            ).sum(-1).mean() * (T * T)
            # hard loss on real next tokens
            ce = F.cross_entropy(
                slog[:, :-1, :].reshape(-1, slog.size(-1)),
                labs[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = (args.alpha * kd + (1 - args.alpha) * ce) / args.accum
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        running += step_loss

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(f"distill step {step+1:5d}/{args.steps}  loss {running/args.log_every:.4f}  "
                  f"lr {lr:.2e}  elapsed {dt/60:.1f}m", flush=True)
            running = 0.0

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            torch.save({"model": {k: v.cpu() for k, v in student.state_dict().items()},
                        "optimizer": opt.state_dict(), "step": step + 1,
                        "config": scfg.to_dict()}, last_path)
            torch.save({"model": {k: v.cpu() for k, v in student.state_dict().items()},
                        "config": scfg.to_dict()},
                       os.path.join(args.out, "model.pt"))
            if args.snapshot_dir and args.snapshot_every and (step + 1) % args.snapshot_every == 0:
                os.makedirs(args.snapshot_dir, exist_ok=True)
                torch.save({"model": {k: v.cpu() for k, v in student.state_dict().items()},
                            "config": scfg.to_dict()},
                           os.path.join(args.snapshot_dir, f"snap_{step+1}.pt"))
            print(f"saved distillation checkpoint at step {step+1}", flush=True)

    # report: how close is the student to the teacher now?
    student.eval()
    with torch.no_grad():
        vals = []
        for _ in range(10):
            b = corpus.sample_batch(args.batch_size, args.seq_len)
            ids = b["input_ids"].to(device)
            labs = b["labels"].to(device)
            vals.append(float(F.cross_entropy(
                student(input_ids=ids)["logits"][:, :-1, :].reshape(-1, scfg.text.vocab_size),
                labs[:, 1:].reshape(-1), ignore_index=-100)))
    with open(os.path.join(args.out, "distill_report.json"), "w") as f:
        json.dump({"steps": args.steps, "final_loss": sum(vals) / len(vals)}, f, indent=2)
    print(f"done. final CE {sum(vals)/len(vals):.4f}", flush=True)


if __name__ == "__main__":
    main()
