"""Self-verification: the model checks its own answer before returning it."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .strict_english import StrictEnglishReport, validate as validate_english


@dataclass
class VerificationResult:
    passed: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    english: Optional[StrictEnglishReport] = None


class Verifier:
    """Combines deterministic checks with optional task-specific checkers.

    The design is intentionally layered so a weaker small model can be
    compensated by strong external checks (unit tests, exact-match, schema).
    """

    def __init__(
        self,
        required_keywords: Optional[List[str]] = None,
        forbidden_regex: Optional[List[str]] = None,
        checker: Optional[Callable[[str], bool]] = None,
        min_english_score: float = 0.85,
        threshold: float = 0.8,
    ) -> None:
        self.required_keywords = required_keywords or []
        self.forbidden_regex = forbidden_regex or []
        self.checker = checker
        self.min_english_score = min_english_score
        self.threshold = threshold

    def verify(self, answer: str) -> VerificationResult:
        reasons: List[str] = []
        score = 1.0

        eng = validate_english(answer)
        if eng.score < self.min_english_score:
            reasons.append(f"strict-English score {eng.score:.2f} below {self.min_english_score}")
        score *= eng.score

        for kw in self.required_keywords:
            if kw.lower() not in answer.lower():
                reasons.append(f"missing required keyword: {kw}")
                score *= 0.8

        for pat in self.forbidden_regex:
            if re.search(pat, answer):
                reasons.append(f"matched forbidden pattern: {pat}")
                score *= 0.5

        if self.checker is not None:
            try:
                ok = bool(self.checker(answer))
            except Exception as e:  # noqa: BLE001
                ok = False
                reasons.append(f"checker raised: {e}")
            if not ok:
                reasons.append("task checker failed")
                score *= 0.5

        passed = score >= self.threshold and not reasons
        return VerificationResult(passed=passed, score=score, reasons=reasons, english=eng)
