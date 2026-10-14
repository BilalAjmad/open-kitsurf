"""Prepare a real tokenizer and a binary token corpus from raw text.

Usage:
    PYTHONPATH=src python3 scripts/prepare_data.py \
        --raw data/raw/tinystories.txt \
        --out-dir data/processed \
        --vocab-size 4096 \
        --tokenizer-sample-mb 24

Writes:
    <out-dir>/tokenizer.json
    <out-dir>/tokens.bin        (uint16, one token per entry)
    <out-dir>/meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from smallm.tokenizer import BPETokenizer

DOC_SEP = "<|endoftext|>"


def iter_documents(path: str, limit_mb: float | None = None):
    """Yield documents split on the TinyStories end-of-text marker."""
    read = 0
    budget = None if limit_mb is None else limit_mb * 1024 * 1024
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            read += len(chunk.encode("utf-8", errors="ignore"))
            buf += chunk
            parts = buf.split(DOC_SEP)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
            if budget is not None and read >= budget:
                break
    if buf.strip():
        yield buf.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=24.0)
    ap.add_argument("--tokenize-limit-mb", type=float, default=None)
    ap.add_argument("--allow-special", action="store_true",
                    help="encode special tokens (<bos>, <user>, ...) as single ids")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok_path = os.path.join(args.out_dir, "tokenizer.json")

    if not os.path.exists(tok_path):
        print(f"[1/3] training BPE vocab={args.vocab_size} on "
              f"{args.tokenizer_sample_mb}MB sample ...")
        sample = list(iter_documents(args.raw, limit_mb=args.tokenizer_sample_mb))
        tok = BPETokenizer.train(sample, vocab_size=args.vocab_size, min_freq=2)
        tok.save(tok_path)
        print(f"      trained vocab size = {len(tok.encoder)}")
    else:
        print(f"[1/3] reusing existing tokenizer {tok_path}")
        tok = BPETokenizer.load(tok_path)

    print("[2/3] tokenizing corpus ...")
    all_ids: list[int] = []
    n_docs = 0
    eos = tok.eos_id
    for doc in iter_documents(args.raw, limit_mb=args.tokenize_limit_mb):
        all_ids.append(tok.bos_id)
        all_ids.extend(tok.encode(doc, allow_special=args.allow_special))
        all_ids.append(eos)
        n_docs += 1
        if n_docs % 20000 == 0:
            print(f"      {n_docs} docs, {len(all_ids)/1e6:.1f}M tokens")

    arr = np.asarray(all_ids, dtype=np.uint16)
    bin_path = os.path.join(args.out_dir, "tokens.bin")
    arr.tofile(bin_path)
    meta = {
        "vocab_size": len(tok.encoder),
        "n_tokens": int(arr.shape[0]),
        "n_docs": n_docs,
        "dtype": "uint16",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[3/3] wrote {bin_path}  tokens={arr.shape[0]/1e6:.1f}M  docs={n_docs}")


if __name__ == "__main__":
    main()
