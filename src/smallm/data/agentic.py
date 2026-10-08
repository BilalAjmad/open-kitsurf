"""Generate agentic fine-tuning traces in the SmallM special-token format.

Each trace teaches the model one behaviour:
  - think before acting           ( thinking ... </think>)
  - call a tool                   (<tool_call>{...}</tool_call>)
  - read the observation          (<tool_result> ... </tool_result>)
  - self-verify                   (<verify> ... </verify>)
  - answer in strict English

The format must match `smallm.agent.model_brain.render_prompt` / `parse_decision`.
"""
from __future__ import annotations

import json
import random
from typing import List

CALC_TASKS = [
    (2, 3, "+"), (7, 5, "-"), (4, 6, "*"), (9, 3, "/"),
    (12, 8, "+"), (20, 4, "/"), (6, 7, "*"), (15, 9, "-"),
]

FILE_TASKS = [
    ("note.txt", "hello"), ("todo.txt", "buy milk"), ("log.txt", "started"),
    ("data.txt", "42"), ("readme.txt", "small model"),
]


def _calc_trace(a: int, b: int, op: str) -> tuple[str, str]:
    expr = f"{a}{op}{b}"
    val = {"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}[op]
    result = json.dumps({"ok": True, "output": str(val), "error": ""})
    text = (
        "<bos><system> Think, then act. Use the calc tool. Verify the result. "
        "Use strict English. </system>"
        f"<user> Compute {expr}. </user><assistant>"
        f" thinkingI compute {expr} with the calc tool.</think>"
        f'<tool_call>{{"name": "calc", "args": {{"expression": "{expr}"}}}}</tool_call>'
        f"<tool_result> {result} </tool_result>"
        f"<verify> The tool returned {val}. That matches my check. </verify>"
        f"The answer is {val}.<eos>"
    )
    return text, f"The answer is {val}."


def _file_trace(path: str, content: str) -> tuple[str, str]:
    write_res = json.dumps({"ok": True, "output": f"wrote {len(content)} chars to {path}", "error": ""})
    read_res = json.dumps({"ok": True, "output": content, "error": ""})
    text = (
        "<bos><system> Think, then act. Use the sandbox tools. Verify the result. "
        "Use strict English. </system>"
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


def _list_trace() -> tuple[str, str]:
    res = json.dumps({"ok": True, "output": "note.txt\ntodo.txt", "error": ""})
    text = (
        "<bos><system> Think, then act. Use the sandbox tools. Verify the result. "
        "Use strict English. </system>"
        "<user> List the files in the sandbox. </user><assistant>"
        " thinkingI list the directory to see the files.</think>"
        '<tool_call>{"name": "list_dir", "args": {"path": "."}}</tool_call>'
        f"<tool_result> {res} </tool_result>"
        "<verify> The listing returned two files. It matches my expectation. </verify>"
        "The sandbox contains note.txt and todo.txt.<eos>"
    )
    return text, "The sandbox contains note.txt and todo.txt."


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


def build_agentic_corpus(seed: int = 0, repeats: int = 3) -> List[str]:
    rng = random.Random(seed)
    docs: List[str] = []
    for _ in range(repeats):
        for a, b, op in CALC_TASKS:
            docs.append(_calc_trace(a, b, op)[0])
        for path, content in FILE_TASKS:
            docs.append(_file_trace(path, content)[0])
        docs.append(_list_trace()[0])
        for e in strict_english_examples():
            docs.append(e)
    rng.shuffle(docs)
    return docs


def build_agentic_sft_pairs(seed: int = 0, repeats: int = 2):
    """Return (prompt, response) pairs for response-only SFT loss."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(repeats):
        for a, b, op in CALC_TASKS:
            full, ans = _calc_trace(a, b, op)
            prompt, resp = full.split("<assistant>", 1)
            pairs.append((prompt + "<assistant>", resp))
        for path, content in FILE_TASKS:
            full, ans = _file_trace(path, content)
            prompt, resp = full.split("<assistant>", 1)
            pairs.append((prompt + "<assistant>", resp))
        full, ans = _list_trace()
        prompt, resp = full.split("<assistant>", 1)
        pairs.append((prompt + "<assistant>", resp))
    rng.shuffle(pairs)
    return pairs
