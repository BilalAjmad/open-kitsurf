"""Turn instruction datasets into a single chat-formatted JSONL corpus.

Reads Databricks Dolly (JSONL, one record per line with instruction/context/
response fields) and writes one chat document per line in the tokenizer's
special-token format:

    <|user|> ... <|assistant|> ... <|endoftext|>

The tokenizer already knows these markers, so no new special tokens are added.

Usage:
    PYTHONPATH=src python3 scripts/build_instruct_corpus.py \
        --raw data/raw/dolly/databricks-dolly-15k.jsonl \
        --out data/curated/instruct.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

USER, ASSISTANT, SYSTEM = "<user>", "<assistant>", "<system>"
THINK_OPEN, THINK_CLOSE = " thinking", "</think>"
DOC_SEP = "<|endoftext|>"


def to_chat(rec: dict) -> str | None:
    instr = (rec.get("instruction") or "").strip()
    ctx = (rec.get("context") or "").strip()
    resp = (rec.get("response") or "").strip()
    if not instr or not resp:
        return None
    user = instr if not ctx else f"{instr}\n\n{ctx}"
    return (f"<bos><system> You are a helpful assistant. </system>"
            f"<user> {user} </user>"
            f"<assistant> {THINK_OPEN}{THINK_CLOSE}{resp} </assistant>"
            f"<eos>" + "\n" + DOC_SEP + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = total = 0
    with open(args.raw, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = to_chat(rec)
            if doc:
                out.write(doc + "\n")
                kept += 1
    print(json.dumps({"raw": total, "kept": kept}))


if __name__ == "__main__":
    main()
