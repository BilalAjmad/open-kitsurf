"""The fully agentic reasoning loop with thinking and self-verification.

Control flow per iteration:
    perceive -> think -> act (tool call) -> observe -> ... -> answer -> verify
If verification fails the loop feeds the failure reasons back into the next
thought, so the model can correct itself before returning.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .strict_english import correct as correct_english
from .tools import ToolRegistry
from .verifier import Verifier


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str


@dataclass
class Step:
    kind: str  # thought | action | observation | answer | verification
    content: str


@dataclass
class AgentResult:
    answer: str
    passed: bool
    steps: List[Step] = field(default_factory=list)
    iterations: int = 0


class Brain(Protocol):
    """Anything that can produce a thought + action or a final answer."""

    def think(self, messages: List[Message], tools: List[dict]) -> Dict[str, Any]:
        ...


class ScriptedBrain:
    """Deterministic brain used for tests and demonstrations.

    `script` is a list of dicts, each either
      {"thought": str, "tool": name, "args": {...}}
    or
      {"thought": str, "answer": str}
    """

    def __init__(self, script: List[Dict[str, Any]]) -> None:
        self.script = list(script)
        self._i = 0

    def think(self, messages: List[Message], tools: List[dict]) -> Dict[str, Any]:
        if self._i >= len(self.script):
            return {"thought": "(no more steps)", "answer": "(no answer)"}
        step = self.script[self._i]
        self._i += 1
        return step


class AgentLoop:
    def __init__(
        self,
        brain: Brain,
        tools: ToolRegistry,
        verifier: Optional[Verifier] = None,
        max_iterations: int = 8,
        max_verify_attempts: int = 3,
    ) -> None:
        self.brain = brain
        self.tools = tools
        self.verifier = verifier or Verifier()
        self.max_iterations = max_iterations
        self.max_verify_attempts = max_verify_attempts

    def run(self, task: str) -> AgentResult:
        messages = [Message("user", task)]
        steps: List[Step] = []
        verify_attempts = 0
        last_action: Optional[str] = None

        for it in range(1, self.max_iterations + 1):
            decision = self.brain.think(messages, self.tools.schemas())
            thought = decision.get("thought", "")
            if thought:
                steps.append(Step("thought", thought))

            if "tool" in decision:
                name = decision["tool"]
                args = decision.get("args", {})
                key = f"{name}({json.dumps(args, sort_keys=True)})"
                if key == last_action:
                    # The model is repeating itself. Do not run the same call
                    # again; demand the prose answer instead.
                    steps.append(Step("thought", "repeated tool call ignored"))
                    messages.append(
                        Message("system", "You already ran " + key
                                + ". Read the observation and write the final answer in prose.")
                    )
                    continue
                last_action = key
                try:
                    result = self.tools.call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = json.dumps({"ok": False, "error": str(e)})
                steps.append(Step("action", f"{name}({json.dumps(args)})"))
                steps.append(Step("observation", result))
                # Describe the call in natural language. Feeding the internal
                # decision JSON back makes small models echo it verbatim.
                messages.append(Message("assistant", f"I called {name} with {json.dumps(args)}."))
                messages.append(Message("tool", result))
                continue

            if "answer" in decision:
                answer = correct_english(decision["answer"])
                # A small model often wraps a correct number in JSON or tool
                # syntax. If the number is one the tools actually produced,
                # present it as clean prose before checking. This rescues the
                # output; it never invents a value.
                observed = _observed_numbers(steps)
                answer = _normalize_answer(answer, observed)
                verdict = self.verifier.verify(answer)
                reasons = list(verdict.reasons)
                passed = verdict.passed

                # An answer is prose. Control tokens or tool-call syntax in it
                # mean the model never actually finished, so reject it.
                leaked = _leaked_control_token(answer)
                if leaked is not None:
                    passed = False
                    reasons.append(f"answer contains control token {leaked}")

                # A JSON object or a function-call fragment is a tool call or an
                # internal record, never a final answer. Reject it so the model
                # produces real prose instead of dumping its bookkeeping.
                if passed and (_looks_like_json(answer) or _looks_like_tool_syntax(answer)):
                    passed = False
                    reasons.append("answer is a tool call or JSON, not prose")

                # Ground the answer in the real tool results. A small model can
                # drift from the observation it just read, so the loop rejects an
                # answer that contradicts a numeric tool output and retries.
                if passed and observed and not any(n in answer for n in observed):
                    passed = False
                    reasons.append(
                        "answer does not match the tool result " + ", ".join(observed)
                    )

                # A task that names a file must not be "answered" before a file
                # tool has run. Otherwise the model can claim success it never did.
                if passed and _task_needs_file_tool(task) and not _used_file_tool(steps):
                    passed = False
                    reasons.append("no file tool has run yet for a file task")

                evidence = _file_evidence(steps)
                if passed and evidence and not any(e in answer for e in evidence):
                    passed = False
                    reasons.append(
                        "answer does not mention the file it worked on "
                        + ", ".join(evidence)
                    )

                steps.append(
                    Step(
                        "verification",
                        f"passed={passed} score={verdict.score:.2f} reasons={reasons}",
                    )
                )
                if passed:
                    steps.append(Step("answer", answer))
                    return AgentResult(answer, True, steps, it)
                verify_attempts += 1
                if verify_attempts >= self.max_verify_attempts:
                    steps.append(Step("answer", answer))
                    return AgentResult(answer, False, steps, it)
                # feed failure back so the brain can self-correct
                messages.append(
                    Message(
                        "system",
                        "Verification failed. Correct these points: " + "; ".join(reasons),
                    )
                )
                continue

            steps.append(Step("thought", "malformed decision, retrying"))
            messages.append(Message("system", "Return either a tool call or an answer."))

        return AgentResult("(max iterations reached)", False, steps, self.max_iterations)


_CONTROL_TOKENS = (
    "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>",
    " thinking", " response", "<verify>", "</verify>", "<eos>", "<bos>",
)


def _leaked_control_token(answer: str) -> Optional[str]:
    for tok in _CONTROL_TOKENS:
        if tok in answer:
            return tok
    return None


def _observed_numbers(steps: List[Step]) -> List[str]:
    """Numeric tool outputs seen so far, e.g. calc's ``"output": "42"``."""
    nums: List[str] = []
    for step in steps:
        if step.kind != "observation":
            continue
        for m in re.finditer(r'"output":\s*"(-?\d+)"', step.content):
            if m.group(1) not in nums:
                nums.append(m.group(1))
    return nums


_NUMBER_RE = re.compile(r"-?\d+")


def _normalize_answer(answer: str, observed: List[str]) -> str:
    """Turn a messy answer that contains a real tool number into clean prose.

    Only fires when the answer is not already good prose and a number in it was
    actually produced by a tool. It never fabricates a value.
    """
    if not observed:
        return answer
    messy = (_leaked_control_token(answer) is not None or _looks_like_json(answer)
             or _looks_like_tool_syntax(answer))
    if not messy:
        return answer
    for n in _NUMBER_RE.findall(answer):
        if n in observed:
            return f"The answer is {n}."
    return answer


def _looks_like_json(answer: str) -> bool:
    s = answer.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        return isinstance(json.loads(s), dict)
    except json.JSONDecodeError:
        return False


def _looks_like_tool_syntax(answer: str) -> bool:
    """Detect a function-call fragment the model dumped instead of prose."""
    return any(k in answer for k in ('"function"', '"parameters"', '"tool_call"',
                                     '"name":', '"args"'))


_FILE_TOOLS = {"write_file", "read_file", "list_dir", "run", "see"}


def _used_file_tool(steps: List[Step]) -> bool:
    for step in steps:
        if step.kind == "action" and step.content.split("(", 1)[0] in _FILE_TOOLS:
            return True
    return False


def _task_needs_file_tool(task: str) -> bool:
    low = task.lower()
    looks_like_file = any(w in low for w in
                          ("file", "write", "read", "create", "note", "directory"))
    looks_like_image = any(w in low for w in (".png", ".jpg", ".jpeg", "image"))
    return looks_like_file or looks_like_image


def _file_evidence(steps: List[Step]) -> List[str]:
    """Paths and contents the file tools actually touched.

    Empty when no file tool ran, so the guard only applies to file tasks.
    """
    evidence: List[str] = []
    for i, step in enumerate(steps):
        if step.kind == "action":
            name = step.content.split("(", 1)[0]
            if name == "see":
                # Grounding for a `see` call is the answer the vision model
                # produced, not the image filename.
                nxt = steps[i + 1].content if i + 1 < len(steps) else ""
                try:
                    out = json.loads(nxt).get("output", "")
                except (json.JSONDecodeError, AttributeError):
                    out = ""
                # Word-level grounding: the answer must share a content word
                # with what the vision model actually saw. Verbatim matching is
                # too strict when the agent rephrases in its own words.
                for w in re.findall(r"[a-z]{4,}", out.lower()):
                    if w not in evidence:
                        evidence.append(w)
                continue
            for m in re.finditer(r'"(?:path|content)":\s*"([^"]+)"', step.content):
                if m.group(1) not in evidence:
                    evidence.append(m.group(1))
        elif step.kind == "observation":
            for m in re.finditer(r"[\w./-]+\.\w+", step.content):
                name = m.group(0).rstrip(".")
                if name not in evidence:
                    evidence.append(name)
    return evidence
