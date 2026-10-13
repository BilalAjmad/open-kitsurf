"""Load real human-written instruction/response pairs (Databricks Dolly-15k).

Each example is rendered into a single prompt/response pair in the plain
`### Instruction / ### Response` format so a small model can learn it.
"""
from __future__ import annotations

import json
from typing import List, Tuple

PROMPT = "### Instruction:\n{instruction}\n### Response:\n"


def format_pair(instruction: str, context: str, response: str) -> Tuple[str, str]:
    inst = instruction.strip()
    if context.strip():
        inst = f"{inst} (context: {context.strip()})"
    return PROMPT.format(instruction=inst), response.strip()


def load_dolly(path: str, limit: int = 0) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            inst = ex.get("instruction", "")
            resp = ex.get("response", "")
            if not inst.strip() or not resp.strip():
                continue
            pairs.append(format_pair(inst, ex.get("context", ""), resp))
            if limit and len(pairs) >= limit:
                break
    return pairs
