"""Build a model soup from several fine-tune checkpoints.

Evaluates each candidate on a held-out slice of the token corpus, then builds
a uniform soup and a greedy soup, keeping whichever has the lowest val loss.

Usage:
    PYTHONPATH=src python3 scripts/make_soup.py \
        --variants checkpoints/var0 checkpoints/var1 checkpoints/var2 \
        --data-dir data/processed \
        --out checkpoints/smallm_11m_soup \
        --val-steps 20 --val-batch 8 --val-seq 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig
from smallm.data import TokenCorpus
from smallm.model import build_model
from smallm.train import uniform_soup, greedy_soup, save_soup


def load_variant(d: str):
    return torch.load(os.path.join(d, "model.pt"), map_location="cpu")


def snapshots_by_step(snap_dir: str, last: int):
    """Return the last `last` training snapshots ordered by step number."""
    import glob
    import re

    files = glob.glob(os.path.join(snap_dir, "snap_*.pt"))
    files.sort(key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
    if last and last > 0:
        files = files[-last:]
    return [torch.load(p, map_location="cpu") for p in files]


def make_eval_loss(cfg, corpus, steps, batch, seq):
    batches = [corpus.sample_batch(batch, seq) for _ in range(steps)]
    model = build_model(cfg)

    def eval_loss(state):
        model.load_state_dict(state)
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in batches:
                out = model(input_ids=b["input_ids"], labels=b["labels"])
                total += float(out["loss"])
        return total / len(batches)

    return eval_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--snapshot-dir", default=None,
                    help="directory of snap_0_<step>.pt checkpoints to soup")
    ap.add_argument("--last", type=int, default=8,
                    help="use only the last N snapshots (0 = all)")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-steps", type=int, default=20)
    ap.add_argument("--val-batch", type=int, default=8)
    ap.add_argument("--val-seq", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    if args.snapshot_dir:
        variants = snapshots_by_step(args.snapshot_dir, args.last)
        names = [f"snapshot[{i}]" for i in range(len(variants))]
    elif args.variants:
        variants = [load_variant(v) for v in args.variants]
        names = args.variants
    else:
        ap.error("pass --variants or --snapshot-dir")
    cfg = ModelConfig.from_dict(variants[0]["config"])
    corpus = TokenCorpus(os.path.join(args.data_dir, "tokens.bin"),
                         os.path.join(args.data_dir, "meta.json"))
    eval_loss = make_eval_loss(cfg, corpus, args.val_steps, args.val_batch, args.val_seq)

    scores = [eval_loss(sd["model"]) for sd in variants]
    for v, s in zip(names, scores):
        print(f"  {v}: val_loss={s:.4f}", flush=True)

    uni = uniform_soup([sd["model"] for sd in variants])
    uni_loss = eval_loss(uni)
    print(f"uniform soup: val_loss={uni_loss:.4f}", flush=True)

    greedy, chosen = greedy_soup([sd["model"] for sd in variants], eval_loss)
    greedy_loss = eval_loss(greedy)
    print(f"greedy soup: val_loss={greedy_loss:.4f} (used {len(chosen)} models)", flush=True)

    best_name, best_sd, best_loss = min(
        [("uniform", uni, uni_loss), ("greedy", greedy, greedy_loss)],
        key=lambda t: t[2],
    )
    save_soup(best_sd, cfg.to_dict(), os.path.join(args.out, "model.pt"))
    tc = os.path.join(args.data_dir, "tokenizer.json")
    if os.path.exists(tc):
        import shutil
        shutil.copy(tc, os.path.join(args.out, "tokenizer.json"))
    report = {
        "variants": names,
        "variant_losses": scores,
        "uniform_loss": uni_loss,
        "greedy_loss": greedy_loss,
        "greedy_chosen": chosen,
        "best": best_name,
    }
    with open(os.path.join(args.out, "soup_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote best soup ({best_name}) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
