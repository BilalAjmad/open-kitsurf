"""Warm-start the Nova architecture from an existing trained SmallM checkpoint.

Nova keeps the same block *contract* as the plain transformer (same projection
names and shapes), so a trained checkpoint can be transplanted: every trained
tensor is copied, and only the Nova-only tensors (q_norm, k_norm, layer scales,
per-step loop embeddings) are freshly initialized. This turns a working model
into a working Nova in seconds instead of a 28-hour from-scratch run; a short
distillation pass then closes the small gap.

Usage:
    python3 scripts/transplant_to_nova.py \
        --teacher checkpoints/smallm_11m/model.pt \
        --config configs/smallm_nova_ready.json \
        --out checkpoints/nova_ready
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig
from smallm.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--config", required=True, help="target Nova config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tck = torch.load(args.teacher, map_location="cpu")
    tsd = tck["model"]
    tcfg = ModelConfig.from_dict(tck["config"]) if "config" in tck else None

    cfg = ModelConfig.load(args.config)
    assert cfg.text.arch == "nova", "target config must be arch=nova"
    model = build_model(cfg)
    sd = model.state_dict()

    copied, new, shape_mismatch = [], [], []
    for k, v in tsd.items():
        if k not in sd:
            # e.g. teacher has no step_embed; nothing teacher-only should exist
            continue
        if sd[k].shape == v.shape:
            sd[k] = v.clone()
            copied.append(k)
        else:
            shape_mismatch.append((k, tuple(v.shape), tuple(sd[k].shape)))

    # Nova-only tensors get a warm start that reproduces the teacher on the
    # first recurrence pass: q/k norms identity, layer scales at 1.0 (so the
    # transplanted branches are actually used, not damped), loop embeddings zero.
    if "text.step_embed" in sd:
        sd["text.step_embed"] = torch.zeros_like(sd["text.step_embed"])
    for k in sd:
        if k.endswith("attn_scale") or k.endswith("ffn_scale"):
            sd[k] = torch.ones_like(sd[k])

    new = [k for k in sd if k not in copied]
    model.load_state_dict(sd)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
               os.path.join(args.out, "model.pt"))
    tc = os.path.join(os.path.dirname(args.teacher), "tokenizer.json")
    if os.path.exists(tc):
        import shutil
        shutil.copy(tc, os.path.join(args.out, "tokenizer.json"))

    n = sum(p.numel() for p in model.parameters())
    print(f"transplanted {len(copied)} tensors into Nova ({n/1e6:.1f}M params)")
    print(f"freshly initialized {len(new)} Nova-only tensors, e.g. {new[:5]}")
    if shape_mismatch:
        print("WARNING shape mismatches (left at init):")
        for k, a, b in shape_mismatch[:10]:
            print(f"  {k}: teacher {a} vs nova {b}")
    if tcfg is not None:
        print(f"teacher: hidden={tcfg.text.hidden_size} layers={tcfg.text.num_hidden_layers}")
    print(f"nova:    hidden={cfg.text.hidden_size} layers={cfg.text.num_hidden_layers} "
          f"x {cfg.text.num_recurrent_steps} loops = depth "
          f"{cfg.text.num_hidden_layers * cfg.text.num_recurrent_steps}")
    print(f"wrote {os.path.join(args.out, 'model.pt')}")


if __name__ == "__main__":
    main()
