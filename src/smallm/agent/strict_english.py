"""Strict-English enforcement for all agent output."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


BANNED_CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "i'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i'll": "I will",
    "we'll": "we will",
    "let's": "let us",
    "that's": "that is",
    "there's": "there is",
}

BANNED_WORDS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "have to",
    "kinda": "kind of",
    "dunno": "do not know",
    "yeah": "yes",
    "nope": "no",
}

MAX_SENTENCE_WORDS = 25


@dataclass
class StrictEnglishReport:
    compliant: bool
    score: float
    corrections: List[str]
    violations: List[str]
    corrected_text: str


def correct(text: str) -> str:
    out = text
    for bad, good in {**BANNED_CONTRACTIONS, **BANNED_WORDS}.items():
        out = re.sub(rf"\b{re.escape(bad)}\b", good, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def validate(text: str) -> StrictEnglishReport:
    violations: List[str] = []
    corrections: List[str] = []
    lowered = text.lower()

    for bad, good in {**BANNED_CONTRACTIONS, **BANNED_WORDS}.items():
        if re.search(rf"\b{re.escape(bad)}\b", lowered):
            violations.append(f"non-strict word: {bad!r}")
            corrections.append(good)

    for s in re.split(r"(?<=[.!?])\s+", text):
        n = len(s.split())
        if n > MAX_SENTENCE_WORDS:
            violations.append(f"sentence too long ({n} words)")

    if "!" in text:
        violations.append("exclamation marks are not permitted")
    if ";" in text:
        violations.append("semicolons are not permitted")

    score = max(0.0, min(1.0, 1.0 - 0.15 * len(violations)))
    return StrictEnglishReport(
        compliant=len(violations) == 0,
        score=score,
        corrections=sorted(set(corrections)),
        violations=violations,
        corrected_text=correct(text),
    )
