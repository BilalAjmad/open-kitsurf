"""The fully agentic reasoning loop with thinking and self-verification.

Control flow per iteration:
    perceive -> think -> act (tool call) -> observe -> ... -> answer -> verify
If verification fails the loop feeds the failure reasons back into the next
thought, so the model can correct itself before returning.
"""
from __future__ import annotations

import json
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

        for it in range(1, self.max_iterations + 1):
            decision = self.brain.think(messages, self.tools.schemas())
            thought = decision.get("thought", "")
            if thought:
                steps.append(Step("thought", thought))

            if "tool" in decision:
                name = decision["tool"]
                args = decision.get("args", {})
                try:
                    result = self.tools.call(name, args)
                except Exception as e:  # noqa: BLE001
                    result = json.dumps({"ok": False, "error": str(e)})
                steps.append(Step("action", f"{name}({json.dumps(args)})"))
                steps.append(Step("observation", result))
                messages.append(Message("assistant", json.dumps(decision)))
                messages.append(Message("tool", result))
                continue

            if "answer" in decision:
                answer = correct_english(decision["answer"])
                verdict = self.verifier.verify(answer)
                steps.append(
                    Step(
                        "verification",
                        f"passed={verdict.passed} score={verdict.score:.2f} reasons={verdict.reasons}",
                    )
                )
                if verdict.passed:
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
                        "Verification failed. Correct these points: "
                        + "; ".join(verdict.reasons),
                    )
                )
                continue

            steps.append(Step("thought", "malformed decision, retrying"))
            messages.append(Message("system", "Return either a tool call or an answer."))

        return AgentResult("(max iterations reached)", False, steps, self.max_iterations)
