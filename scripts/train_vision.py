"""Train the vision tower on a synthetic image -> caption task.

The generator draws one coloured shape on a black canvas. The caption names the
colour and shape, so the loss can only fall if the ViT, the resampler, and the
projector actually read the pixels. This is a real, measurable vision signal on
CPU; it is not a stand-in for a full vision-language corpus.

Usage:
    PYTHONPATH=src python3 scripts/train_vision.py \
        --init checkpoints/smallm_11m/model.pt \
        --out checkpoints/smallm_11m_vision \
        --steps 2000 --batch-size 16
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from smallm.config import ModelConfig
from smallm.model import build_model
from smallm.tokenizer import BPETokenizer
from smallm.train import make_optimizer

COLOURS = ["red", "green", "blue", "yellow", "white"]
SHAPES = ["circle", "square", "triangle"]

# colour id and shape id encoded as pixel values, so the model must see them.
_N_IMG_TOKENS = 16


_RGB = {
    0: (0.90, 0.10, 0.10),  # red
    1: (0.10, 0.80, 0.10),  # green
    2: (0.10, 0.20, 0.90),  # blue
    3: (0.90, 0.90, 0.10),  # yellow
    4: (0.95, 0.95, 0.95),  # white
}
_N = 64


def _shape_mask(si: int) -> "np.ndarray":
    """Boolean mask for the shape, in a centred disk of radius ~22."""
    c = _N // 2
    yy, xx = np.mgrid[0:_N, 0:_N]
    dx, dy = xx - c, yy - c
    r = 22
    if si == 0:  # circle
        return dx * dx + dy * dy <= r * r
    if si == 1:  # square
        return (np.abs(dx) <= r) & (np.abs(dy) <= r)
    # triangle: apex at top, base at bottom, widening with depth
    t = (dy + r) / (2 * r)
    return (t >= 0) & (t <= 1) & (np.abs(dx) <= t * r)


def make_batch(batch_size: int, image_size: int, rng: torch.Generator):
    """Return (images, captions) for a batch of random coloured-shape images."""
    del image_size  # the mask is authored at 64px; images are always 64px here
    imgs = torch.zeros(batch_size, 3, _N, _N)
    captions = []
    for i in range(batch_size):
        ci = int(torch.randint(0, len(COLOURS), (1,), generator=rng).item())
        si = int(torch.randint(0, len(SHAPES), (1,), generator=rng).item())
        mask = _shape_mask(si)
        imgs[i][:, mask] = torch.tensor(_RGB[ci]).view(3, 1)
        captions.append(f"This is a {COLOURS[ci]} {SHAPES[si]}.")
    return imgs, captions


def encode_batch(captions, tok, n_img_tokens, seq_len):
    rows, labs = [], []
    for cap in captions:
        body = tok.encode(cap, allow_special=True) + [tok.eos_id]
        ids = [tok.bos_id] + [3] * n_img_tokens + body
        lab = [-100] * (1 + n_img_tokens) + body
        ids, lab = ids[:seq_len], lab[:seq_len]
        while len(ids) < seq_len:
            ids.append(tok.pad_id)
            lab.append(-100)
        rows.append(ids)
        labs.append(lab)
    return torch.tensor(rows), torch.tensor(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--config", default="configs/smallm_11m.json")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = ModelConfig.load(args.config)
    model = build_model(cfg)
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"initialised from {args.init}")
    model.train()
    opt = make_optimizer(model, lr=args.lr)
    tok = BPETokenizer.load(args.tokenizer)
    n_img = model.vision.num_visual_tokens
    seq_len = 1 + n_img + 16
    rng = torch.Generator().manual_seed(args.seed)
    image_size = cfg.vision.image_size
    print(f"vision SFT on {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{n_img} visual tokens, image {image_size}px")

    for step in range(args.steps):
        imgs, caps = make_batch(args.batch_size, image_size, rng)
        ids, labs = encode_batch(caps, tok, n_img, seq_len)
        out = model(input_ids=ids, images=imgs, labels=labs)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            print(f"vision step {step+1}/{args.steps}  loss {out['loss'].item():.4f}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            _save(model, cfg, tok, args.out, f"vision_{step+1}.pt")
    _save(model, cfg, tok, args.out, "model.pt")
    print("done. wrote vision checkpoint")


def _save(model, cfg, tok, out_dir, name):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, os.path.join(out_dir, name))
    tok.save(os.path.join(out_dir, "tokenizer.json"))


if __name__ == "__main__":
    main()
