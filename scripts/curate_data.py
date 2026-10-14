"""Quality-first corpus curation for the Chimera pretraining run.

This is the "new way" of preparing data for this repo: instead of dumping raw
documents into the tokenizer, every document is scored and filtered, and
near-duplicates are removed with a shingle sketch. The output is a smaller,
cleaner corpus that trains a small model further per token.

Stages, in order:

1. **Normalise** — strip control characters, collapse runaway whitespace.
2. **Hard filters** — length bounds, alphabetic ratio, punctuation presence,
   a cap on the longest word (catches code/markup dumps).
3. **Repetition filter** — reject documents whose most common trigram covers
   too much of the text (catches loops and spam).
4. **Near-dedup** — a 64-bit MinHash-style sketch over word 5-grams; a document
   is dropped if its sketch collides with one already kept. This removes
   paraphrases that exact hashing misses, at a tiny memory cost.

Usage:
    PYTHONPATH=src python3 scripts/curate_data.py \
        --raw data/raw/tinystories.txt \
        --out data/curated/chimera.jsonl \
        --min-words 40 --max-words 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DOC_SEP = "<|endoftext|>"
_WS = re.compile(r"[ \t\r\f\v]+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def iter_documents(path: str):
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(DOC_SEP)
            buf = parts.pop()
            for p in parts:
                yield p
    if buf.strip():
        yield buf


def normalise(text: str) -> str:
    text = _CTRL.sub("", text)
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def trigram_repetition(words) -> float:
    if len(words) < 3:
        return 1.0
    tri = Counter(zip(words, words[1:], words[2:]))
    return tri.most_common(1)[0][1] / (len(words) - 2)


def shingle_sketch(words, k: int = 5, n_hashes: int = 4) -> tuple:
    """Cheap MinHash-style sketch: min hash under several salts."""
    if len(words) < k:
        payload = " ".join(words).encode("utf-8")
        return (hashlib.blake2b(payload, digest_size=8).digest(),) * n_hashes
    grams = [" ".join(words[i:i + k]).encode("utf-8") for i in range(len(words) - k + 1)]
    sketch = []
    for salt in range(n_hashes):
        h = min(hashlib.blake2b(bytes([salt]) + g, digest_size=8).digest() for g in grams)
        sketch.append(h)
    return tuple(sketch)


def quality_ok(text: str, min_words: int, max_words: int,
               min_alpha: float, max_word_len: int,
               max_tri_rep: float):
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False, words
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(1, len(text)) < min_alpha:
        return False, words
    if max((len(w) for w in words), default=0) > max_word_len:
        return False, words
    if not any(c in ".!?" for c in text):
        return False, words
    if trigram_repetition(words) > max_tri_rep:
        return False, words
    return True, words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--min-alpha", type=float, default=0.80)
    ap.add_argument("--max-word-len", type=int, default=24)
    ap.add_argument("--max-tri-rep", type=float, default=0.05)
    ap.add_argument("--report", default=None, help="write a JSON stats file")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    seen: set = set()
    stats = {"raw": 0, "kept": 0, "too_short_or_long": 0, "low_alpha": 0,
             "long_word": 0, "no_punct": 0, "repetitive": 0, "duplicate": 0}
    kept_words = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for raw in iter_documents(args.raw):
            stats["raw"] += 1
            text = normalise(raw)
            if not text:
                continue
            words = text.split()
            if not (args.min_words <= len(words) <= args.max_words):
                stats["too_short_or_long"] += 1
                continue
            letters = sum(c.isalpha() or c.isspace() for c in text)
            if letters / max(1, len(text)) < args.min_alpha:
                stats["low_alpha"] += 1
                continue
            if max((len(w) for w in words), default=0) > args.max_word_len:
                stats["long_word"] += 1
                continue
            if not any(c in ".!?" for c in text):
                stats["no_punct"] += 1
                continue
            if trigram_repetition(words) > args.max_tri_rep:
                stats["repetitive"] += 1
                continue
            sketch = shingle_sketch(words)
            if sketch in seen:
                stats["duplicate"] += 1
                continue
            seen.add(sketch)
            out.write(text + "\n" + DOC_SEP + "\n")
            stats["kept"] += 1
            kept_words += len(words)

    stats["kept_words"] = kept_words
    print(json.dumps(stats, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
