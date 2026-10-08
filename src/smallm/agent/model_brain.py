"""Bridge between the SmallM language model and the agent loop.

The model emits the same special tokens the tokenizer defines:
     thinking ... </think>                  hidden reasoning
    <tool_call>{"name":..,"args":{..}}</tool_call>
    <tool_result> ... </tool_result>          (produced by the loop)
    <verify> ... </verify>                  internal self-check
The brain parses a generated string into the decision dict the loop expects.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch

from .loop import Brain, Message

SYSTEM_PROMPT = (
    "You are SmallM, a strict-English agent. "
    "Think inside  thinking and </think>. "
    "Call a tool with <tool_call>{\"name\": ..., \"args\": {...}}</tool_call>. "
    "Check your own work inside <verify> and </verify>. "
    "Use short sentences. Do not use contractions. "
    "Never use exclamation marks or semicolons. "
    "Answer only when the work is verified."
)

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r" thinking(.*?)</think>", re.DOTALL)
_VERIFY_RE = re.compile(r"<verify>(.*?)</verify>", re.DOTALL)


def render_prompt(messages: List[Message], tools: List[dict]) -> str:
    lines = [f"<bos><system> {SYSTEM_PROMPT}"]
    if tools:
        lines.append("Tools: " + json.dumps(tools))
    for m in messages:
        lines.append(f"<{m.role}> {m.content}")
    lines.append("<assistant>")
    return "\n".join(lines)


def parse_decision(text: str) -> Dict[str, Any]:
    decision: Dict[str, Any] = {}
    think = _THINK_RE.search(text)
    if think:
        decision["thought"] = think.group(1).strip()
    elif text.strip():
        decision["thought"] = text.strip()

    tool = _TOOL_RE.search(text)
    if tool:
        try:
            payload = json.loads(tool.group(1))
            decision["tool"] = payload["name"]
            decision["args"] = payload.get("args", {})
            return decision
        except (json.JSONDecodeError, KeyError):
            return {"thought": "failed to parse tool call"}

    # If the model produced a verify block but no tool, strip it from the answer.
    answer = _VERIFY_RE.sub("", text)
    answer = _THINK_RE.sub("", answer).strip()
    if answer:
        decision["answer"] = answer
    return decision


class ModelBrain(Brain):
    def __init__(self, model, tokenizer, max_new_tokens: int = 256, temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def think(self, messages: List[Message], tools: List[dict]) -> Dict[str, Any]:
        prompt = render_prompt(messages, tools)
        ids = torch.tensor([self.tokenizer.encode(prompt, allow_special=True)])
        out = self.model.generate(
            ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_id,
            temperature=self.temperature,
        )
        gen_ids = out[0, ids.shape[1]:].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special=True)
        return parse_decision(text)
