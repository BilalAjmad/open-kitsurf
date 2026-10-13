"""Real vision eval: a 256M vision-language model answers questions about images.

Generates coloured-shape images with known labels, asks the model, and scores.
This is real image understanding from pixels, not a synthetic caption head.

    PYTHONPATH=.pylibs:src python3 scripts/vision_real.py --n 12
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

COLOURS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (40, 80, 220),
           "yellow": (240, 220, 40), "purple": (150, 60, 200)}
SHAPES = ("circle", "square", "triangle")


def make_image(colour: str, shape: str, size: int = 256):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    fill = COLOURS[colour]
    m = size // 4
    box = [m, m, size - m, size - m]
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    else:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=fill)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="number of images")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from smallm.agent import VLMBrain

    print(f"loading {args.model} ...", flush=True)
    brain = VLMBrain(args.model, max_new_tokens=args.max_new_tokens)
    print(f"params {brain.num_params / 1e6:.1f}M", flush=True)

    rng = random.Random(args.seed)
    colour_ok = shape_ok = 0
    for i in range(args.n):
        colour = rng.choice(list(COLOURS))
        shape = rng.choice(SHAPES)
        img = make_image(colour, shape)
        c = brain.ask(img, "What colour is the shape? Answer with one word.")
        s = brain.ask(img, "What shape is it? Answer with one word.")
        c_hit = colour in c.answer.lower()
        s_hit = shape in s.answer.lower()
        colour_ok += c_hit
        shape_ok += s_hit
        print(f"[{i+1:>2}] colour={colour:<6} shape={shape:<8} "
              f"-> {c.answer!r} {'OK' if c_hit else 'X'} | "
              f"{s.answer!r} {'OK' if s_hit else 'X'}", flush=True)

    print(f"\ncolour accuracy {colour_ok}/{args.n}")
    print(f"shape  accuracy {shape_ok}/{args.n}")


if __name__ == "__main__":
    main()
