"""Export a ready-to-use model.pt from a training `last.pt` checkpoint.

`train_real.py` only writes `model.pt` when it reaches the final step. This lets
you turn any intermediate `last.pt` (weights + optimizer) into the weights-only
artifact that `run.py` and the benchmark load.

    python3 scripts/export_ckpt.py checkpoints/ageis_15m
"""
from __future__ import annotations

import os
import shutil
import sys

import torch


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ageis_15m"
    last = os.path.join(out, "last.pt")
    sd = torch.load(last, map_location="cpu")
    torch.save({"model": sd["model"], "config": sd["config"]},
               os.path.join(out, "model.pt"))
    tok = os.path.join("data", "processed", "tokenizer.json")
    if os.path.exists(tok) and not os.path.exists(os.path.join(out, "tokenizer.json")):
        shutil.copy(tok, os.path.join(out, "tokenizer.json"))
    print(f"exported {out}/model.pt from step {sd.get('step')}")


if __name__ == "__main__":
    main()
