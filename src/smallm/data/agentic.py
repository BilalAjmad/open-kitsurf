"""Generate agentic fine-tuning traces in the SmallM special-token format.

Each trace teaches the model one behaviour:
  - think before acting           ( thinking ... </think>)
  - call a tool                   (<tool_call>{...}</tool_call>)
  - read the observation          (<tool_result> ... </tool_result>)
  - self-verify                   (<verify> ... </verify>)
  - answer in strict English

Traces are generated over a large task space with randomised operands, paths and
contents. A small model only learns to copy the user input into the tool call if
it sees many distinct inputs, so the generator samples widely instead of
repeating a handful of fixed examples.

The format must match `smallm.agent.model_brain.render_prompt` / `parse_decision`.
"""
from __future__ import annotations

import json
import random
from typing import List, Tuple

SYSTEM_PROMPT = (
    "Think, then act. Verify the result. Use strict English. "
    "Tools: calc, write_file, read_file, list_dir, run, screenshot."
)

_CALC_VERBS = ["Compute", "Calculate", "What is", "Find", "Work out"]
_CALC_THINK = [
    "I compute {expr} with the calc tool.",
    "I use the calc tool for {expr}.",
    "I call calc to evaluate {expr}.",
]
_CALC_VERIFY = [
    "The tool returned {val}. That matches my check.",
    "The result is {val}. My check agrees.",
    "The calc tool gave {val}. This is correct.",
]

_FILE_PATHS = [
    "note.txt", "todo.txt", "log.txt", "data.txt", "readme.txt",
    "plan.txt", "list.txt", "memo.txt", "draft.txt", "out.txt",
]
_FILE_CONTENTS = [
    "hello", "buy milk", "started", "42", "small model", "done",
    "hello world", "ship it", "test", "ok",
]


def _calc(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a // b if b else 0


def _calc_trace(a: int, b: int, op: str, rng: random.Random) -> Tuple[str, str]:
    expr = f"{a}{op}{b}"
    val = _calc(a, b, op)
    result = json.dumps({"ok": True, "output": str(val), "error": ""})
    verb = rng.choice(_CALC_VERBS)
    ask = f"{verb} {expr}?" if verb == "What is" else f"{verb} {expr}."
    think = rng.choice(_CALC_THINK).format(expr=expr)
    verify = rng.choice(_CALC_VERIFY).format(val=val)
    text = (
        f"<bos><system> {SYSTEM_PROMPT} </system>"
        f"<user> {ask} </user><assistant>"
        f" thinking{think}</think>"
        f'<tool_call>{{"name": "calc", "args": {{"expression": "{expr}"}}}}</tool_call>'
        f"<tool_result> {result} </tool_result>"
        f"<verify> {verify} </verify>"
        f"The answer is {val}.<eos>"
    )
    return text, f"The answer is {val}."


def _file_trace(path: str, content: str) -> Tuple[str, str]:
    write_res = json.dumps(
        {"ok": True, "output": f"wrote {len(content)} chars to {path}", "error": ""}
    )
    read_res = json.dumps({"ok": True, "output": content, "error": ""})
    text = (
        f"<bos><system> {SYSTEM_PROMPT} </system>"
        f"<user> Create {path} with the content {content}. Then check it. </user><assistant>"
        f" thinkingI write the file first.</think>"
        f'<tool_call>{{"name": "write_file", "args": {{"path": "{path}", "content": "{content}"}}}}</tool_call>'
        f"<tool_result> {write_res} </tool_result>"
        f" thinkingNow I read the file to verify its content.</think>"
        f'<tool_call>{{"name": "read_file", "args": {{"path": "{path}"}}}}</tool_call>'
        f"<tool_result> {read_res} </tool_result>"
        f"<verify> The read content equals {content}. The write is correct. </verify>"
        f"I wrote {path}. The file contains {content}.<eos>"
    )
    return text, f"I wrote {path}. The file contains {content}."


def _list_trace(files: List[str]) -> Tuple[str, str]:
    listing = "\n".join(files)
    res = json.dumps({"ok": True, "output": listing, "error": ""})
    joined = " and ".join(files)
    text = (
        f"<bos><system> {SYSTEM_PROMPT} </system>"
        "<user> List the files in the sandbox. </user><assistant>"
        " thinkingI list the directory to see the files.</think>"
        '<tool_call>{"name": "list_dir", "args": {"path": "."}}</tool_call>'
        f"<tool_result> {res} </tool_result>"
        f"<verify> The listing returned {len(files)} files. It matches my expectation. </verify>"
        f"The sandbox contains {joined}.<eos>"
    )
    return text, f"The sandbox contains {joined}."


def strict_english_examples() -> List[str]:
    return [
        "<bos><user> Say a short greeting in strict English. </user><assistant>"
        "Hello. I am ready to help. What do you need?<eos>",
        "<bos><user> Explain what a tool call is in strict English. </user><assistant>"
        "A tool call is a request. The model asks a tool to do one task. "
        "The tool returns a result. The model reads the result.<eos>",
        "<bos><user> Describe the agent loop in strict English. </user><assistant>"
        "The agent thinks first. Then it acts. Then it observes the result. "
        "Then it checks its work. Then it answers.<eos>",
    ]


def build_agentic_sft_pairs(
    seed: int = 0, repeats: int = 1, n_calc: int = 320, n_file: int = 60
):
    """Return (prompt, response) pairs for response-only SFT loss."""
    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    for _ in range(repeats):
        for _ in range(n_calc):
            a = rng.randint(1, 99)
            b = rng.randint(1, 12)
            op = rng.choice(["+", "-", "*", "/"])
            full, _ = _calc_trace(a, b, op, rng)
            prompt, resp = full.split("<assistant>", 1)
            pairs.append((prompt + "<assistant>", resp))
        for _ in range(n_file):
            path = rng.choice(_FILE_PATHS)
            content = rng.choice(_FILE_CONTENTS)
            full, _ = _file_trace(path, content)
            prompt, resp = full.split("<assistant>", 1)
            pairs.append((prompt + "<assistant>", resp))
        files = rng.sample(_FILE_PATHS, rng.randint(2, 4))
        full, _ = _list_trace(files)
        prompt, resp = full.split("<assistant>", 1)
        pairs.append((prompt + "<assistant>", resp))
    rng.shuffle(pairs)
    return pairs


def build_agentic_corpus(seed: int = 0, repeats: int = 3) -> List[str]:
    """Full documents (prompt + response) for continued pretraining on the format."""
    rng = random.Random(seed)
    docs: List[str] = []
    for _ in range(repeats):
        for _ in range(80):
            a = rng.randint(1, 30)
            b = rng.randint(1, 12)
            op = rng.choice(["+", "-", "*", "/"])
            docs.append(_calc_trace(a, b, op, rng)[0])
        for _ in range(20):
            docs.append(_file_trace(rng.choice(_FILE_PATHS), rng.choice(_FILE_CONTENTS))[0])
        docs.append(_list_trace(rng.sample(_FILE_PATHS, rng.randint(2, 4)))[0])
        docs.extend(strict_english_examples())
    rng.shuffle(docs)
    return docs
