"""Response-only SFT that teaches a pretrained checkpoint the agentic format.

Turns a plain story model into a thinking, tool-calling, self-verifying agent.
Loss is computed only on the assistant response, so the model is not trained to
reproduce the prompt.

Usage:
    PYTHONPATH=src python3 scripts/sft_agentic.py \
        --init checkpoints/smallm_30m/model.pt \
        --tokenizer data/processed/tokenizer.json \
        --out checkpoints/smallm_30m_agent \
        --steps 600 --batch-size 8 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig
from smallm.data import build_agentic_sft_pairs
from smallm.model import build_model
from smallm.tokenizer import BPETokenizer
from smallm.train import make_optimizer


def encode_pair(prompt: str, response: str, tok: BPETokenizer, seq_len: int,
                train_on_prompt: bool = False):
    p = tok.encode(prompt, allow_special=True)
    r = tok.encode(response, allow_special=True)
    ids = (p + r)[:seq_len]
    prefix = p if train_on_prompt else [-100] * len(p)
    labels = (prefix + r)[:seq_len]
    return ids, labels


def pad_batch(pairs, tok, seq_len, train_on_prompt: bool = False):
    rows, labs = [], []
    for prompt, resp in pairs:
        ids, lab = encode_pair(prompt, resp, tok, seq_len, train_on_prompt)
        while len(ids) < seq_len:
            ids.append(tok.pad_id)
            lab.append(-100)
        rows.append(ids)
        labs.append(lab)
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n-calc", type=int, default=320)
    ap.add_argument("--n-file", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-on-prompt", action="store_true",
                    help="also compute loss on prompt tokens (full-sequence loss)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    sd = torch.load(args.init, map_location="cpu")
    cfg = ModelConfig.from_dict(sd["config"])
    model = build_model(cfg)
    model.load_state_dict(sd["model"])
    tok = BPETokenizer.load(args.tokenizer)
    print(f"SFT on {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    pairs = build_agentic_sft_pairs(
        seed=args.seed, repeats=args.repeats, n_calc=args.n_calc, n_file=args.n_file
    )
    print(f"agentic SFT pairs = {len(pairs)}", flush=True)
    opt = make_optimizer(model, lr=args.lr)
    model.train()

    for step in range(args.steps):
        batch_pairs = [pairs[(step * args.batch_size + i) % len(pairs)]
                       for i in range(args.batch_size)]
        batch = pad_batch(batch_pairs, tok, args.seq_len, args.train_on_prompt)
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"sft step {step+1}/{args.steps}  loss {float(out['loss'].detach()):.4f}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                       os.path.join(args.out, f"sft_{step+1}.pt"))

    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(args.out, "model.pt"))
    import shutil
    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    print("done. wrote agent SFT checkpoint", flush=True)


if __name__ == "__main__":
    main()
