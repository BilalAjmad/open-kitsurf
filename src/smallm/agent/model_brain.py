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

from ..data.agentic import SYSTEM_PROMPT
from .loop import Brain, Message

_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r" thinking(.*?)</think>", re.DOTALL)
_VERIFY_RE = re.compile(r"<verify>(.*?)</verify>", re.DOTALL)


def render_prompt(messages: List[Message], tools: List[dict]) -> str:
    """Rebuild the exact token layout the agentic SFT traces use.

    The layout is fixed: one <system> turn, then alternating <user> turns,
    <assistant> turns that contain a thought and optional tool call, and
    <tool_result> turns. The model was fine-tuned on this layout, so inference
    must reproduce it byte for byte.
    """
    parts = [f"<bos><system> {SYSTEM_PROMPT} </system>"]
    for m in messages:
        if m.role == "tool":
            parts.append(f"<tool_result> {m.content} </tool_result>")
        elif m.role == "assistant":
            try:
                d = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                parts.append(f"<assistant> {m.content}")
                continue
            seg = f"<assistant>  thinking{d.get('thought', '')}</think>"
            if "tool" in d:
                call = {"name": d["tool"], "args": d.get("args", {})}
                seg += f"<tool_call>{json.dumps(call)}</tool_call>"
            parts.append(seg)
        else:
            parts.append(f"<{m.role}> {m.content} </{m.role}>")
    parts.append("<assistant>")
    return "".join(parts)


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
            # "arguments" is what Qwen/Llama chat templates emit; "args" is ours.
            decision["args"] = payload.get("args", payload.get("arguments", {})) or {}
            return decision
        except (json.JSONDecodeError, KeyError):
            return {"thought": "failed to parse tool call"}

    # If the model produced a verify block but no tool, strip it from the answer.
    answer = _VERIFY_RE.sub("", text)
    answer = _THINK_RE.sub("", answer)
    for tok in ("<bos>", "<eos>", "<pad>", "<assistant>", "<user>", "<system>"):
        answer = answer.replace(tok, "")
    answer = answer.strip()
    m = re.search(r"The answer is\s*(-?\d+)", answer)
    if m:
        answer = f"The answer is {m.group(1)}."
    if answer:
        decision["answer"] = answer
    return decision


class ModelBrain(Brain):
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Gentle penalties: a strong repetition penalty pushes the model away
        # from the exact tokens it must copy from the user turn.
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    @torch.no_grad()
    def think(self, messages: List[Message], tools: List[dict]) -> Dict[str, Any]:
        prompt = render_prompt(messages, tools)
        ids = torch.tensor([self.tokenizer.encode(prompt, allow_special=True)])
        out = self.model.generate(
            ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_id,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        gen_ids = out[0, ids.shape[1]:].tolist()
        # keep special tokens so parse_decision can see  thinking/<tool_call>/<verify>
        text = self.tokenizer.decode(gen_ids, skip_special=False)
        return parse_decision(_truncate_at_boundary(text))


def _truncate_at_boundary(text: str) -> str:
    """Cut off text the model must not generate itself.

    A small model can drift after the turn it meant to produce. The loop owns
    <tool_result> and <eos>, and an assistant turn never continues past
    </tool_call>. Cutting at the earliest boundary keeps the parse clean.
    """
    cut = len(text)
    for marker in ("<tool_result>", "<eos>", "</tool_call>"):
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i + len(marker))
    return text[:cut]
