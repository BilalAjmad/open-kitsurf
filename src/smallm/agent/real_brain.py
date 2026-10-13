"""A Brain backed by a real GGUF model (llama.cpp) driving the agent loop.

This is how a small *real* model (0.5B-1B) becomes genuinely agentic on a CPU
box: the model supplies language and reasoning, the loop supplies tools,
observation grounding, and strict-English enforcement. The model is small; the
system around it is not.

Only the `llama_cpp.Llama` object is required, so any GGUF loads.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .loop import Message
from .model_brain import parse_decision

SYSTEM = """You are SmallM, a strictly agentic AI that runs on its own computer.

Write in strict English: short sentences, active voice, no contractions, no \
exclamation marks.

You work in a loop. Each reply is exactly one of two things.

1. A tool call, in exactly this form:
<tool_call>{"name": "<tool>", "args": {...}}</tool_call>

2. A final answer. For arithmetic, end with exactly:
The answer is <number>.
For any other task, write one short sentence.

Rules:
- Use a tool to compute, read, write, or list. Never guess a number; use calc.
- Read the observation after each tool call. Check it. If it is wrong, call the tool again.
- Give the final answer only when it is correct.
- Do not mix a tool call and a final answer in one reply.
"""


def _to_openai_tools(tools: List[dict]) -> List[dict]:
    """Convert the repo's tool schemas to OpenAI function-calling schemas."""
    out = []
    for t in tools:
        props = {k: {"type": "string" if v == "str" else "integer"}
                 for k, v in t["parameters"].items()}
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object", "properties": props,
                               "required": list(props)},
            },
        })
    return out


def _to_chat(messages: List[Message]) -> List[Dict[str, str]]:
    chat = [{"role": "system", "content": SYSTEM}]
    for m in messages:
        if m.role == "tool":
            chat.append({"role": "user", "content": f"Observation: {m.content}"})
        elif m.role == "assistant":
            chat.append({"role": "assistant", "content": m.content})
        elif m.role == "system":
            chat.append({"role": "system", "content": m.content})
        else:
            chat.append({"role": "user", "content": m.content})
    return chat


class RealBrain:
    """Drives a GGUF model through the think -> act -> observe -> answer loop."""

    def __init__(self, llm, max_new_tokens: int = 400, temperature: float = 0.2):
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def think(self, messages: List[Message], tools: List[dict]) -> Dict[str, Any]:
        out = self.llm.create_chat_completion(
            messages=_to_chat(messages),
            tools=_to_openai_tools(tools),
            tool_choice="auto",
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls")
        if calls:
            fn = calls[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return {"thought": "calling a tool", "tool": fn["name"], "args": args}
        text = (msg.get("content") or "").strip()
        # Some models put a JSON tool call in content instead of tool_calls.
        bare = _try_bare_json(text)
        if bare is not None:
            return bare
        decision = parse_decision(text)
        if "tool" not in decision and "answer" not in decision:
            decision = {"thought": "answered", "answer": text}
        return decision


def _try_bare_json(text: str) -> Dict[str, Any] | None:
    """Recover a tool call a model wrote as a bare JSON object.

    Qwen uses ``arguments``, Llama uses ``parameters``, this repo uses ``args``.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    # "name" is the OpenAI/Qwen shape; "tool" is this repo's own decision shape,
    # which the model sometimes echoes from the conversation history.
    name = payload.get("name") or payload.get("tool")
    if not name:
        return None
    args = payload.get("args") or payload.get("arguments") or payload.get("parameters") or {}
    if not isinstance(args, dict):
        return None
    return {"thought": "calling a tool", "tool": name, "args": args}


def load_llama(gguf_path: str, n_ctx: int = 4096, n_threads: int = 4):
    from llama_cpp import Llama

    return Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
