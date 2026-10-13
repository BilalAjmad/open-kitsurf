"""A compact byte-level BPE tokenizer with agentic special tokens.

Byte-level fallback guarantees lossless round-tripping for any input, which is
important for a strict-English agent that must never silently mangle a string.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

# GPT-2-style pretokenizer: a word absorbs its single leading space, so the
# model predicts " the" as one token instead of a bare space followed by "the".
# Without this, whitespace becomes the single most frequent token and the model
# can collapse into space loops. The same function runs at train and inference
# time, so encoding stays lossless and consistent.
_PRETOK_RE = re.compile(r" ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z0-9]+|\s+")


def _pretokenize(text: str) -> List[str]:
    return _PRETOK_RE.findall(text)

SPECIAL_TOKENS = [
    "<pad>",          # 0
    "<bos>",          # 1
    "<eos>",          # 2
    "<image>",        # 3
    " thinking",       # 4
    "</think>",       # 5
    "<verify>",       # 6
    "</verify>",      # 7
    "<tool_call>",    # 8
    "</tool_call>",   # 9
    "<tool_result>",  # 10
    "</tool_result>", # 11
    "<computer>",     # 12
    "</computer>",    # 13
]


def _bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
# whitespace bytes (space, tab, newline, CR, VT, FF) in the byte->unicode space
_WS_CHARS = {BYTE_TO_UNICODE[b] for b in (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C)}


def _get_pairs(word: Tuple[str, ...]) -> set:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class BPETokenizer:
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]], specials=None):
        self.encoder = dict(vocab)
        self.decoder = {i: s for s, i in vocab.items()}
        self.merges = [tuple(m) for m in merges]
        self.bpe_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self._cache: Dict[str, List[int]] = {}

        self.specials = specials or SPECIAL_TOKENS
        self.special_to_id = {t: self.encoder[t] for t in self.specials}
        self.pad_id = self.special_to_id["<pad>"]
        self.bos_id = self.special_to_id["<bos>"]
        self.eos_id = self.special_to_id["<eos>"]
        self.image_token_id = self.special_to_id["<image>"]

    # ---- training -------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 4096, min_freq: int = 2):
        word_freq: Counter = Counter()
        for t in texts:
            for word in _pretokenize(t):
                word_freq[tuple(BYTE_TO_UNICODE[b] for b in word.encode("utf-8"))] += 1

        vocab: Dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            vocab[tok] = len(vocab)
        for b in range(256):
            u = BYTE_TO_UNICODE[b]
            if u not in vocab:
                vocab[u] = len(vocab)

        merges: List[Tuple[str, str]] = []
        words: Dict[Tuple[str, ...], int] = {}
        for w, f in word_freq.items():
            if f >= min_freq:
                words[w] = words.get(w, 0) + f

        # Incremental pair index: counts[pair] = total frequency, and
        # where[pair][word] = frequency, so a merge updates only the words
        # that actually contain the merged pair instead of rescanning all text.
        counts: Counter = Counter()
        where: Dict[Tuple[str, str], Dict[Tuple[str, ...], int]] = {}
        for w, f in words.items():
            for p in _get_pairs(w):
                counts[p] += f
                where.setdefault(p, {})
                where[p][w] = where[p].get(w, 0) + f

        while len(vocab) < vocab_size and counts:
            best, freq = max(counts.items(), key=lambda kv: kv[1])
            if freq < min_freq:
                break
            # never merge whitespace into longer runs (keeps space a single token)
            if all(c in _WS_CHARS for c in best):
                counts.pop(best, None)
                continue
            new_tok = best[0] + best[1]
            if new_tok in vocab:
                break
            vocab[new_tok] = len(vocab)
            merges.append(best)

            for w in list(where.get(best, {}).keys()):
                f = words.pop(w, None)
                if f is None:
                    continue
                for p in _get_pairs(w):
                    counts[p] -= f
                    if counts[p] <= 0:
                        counts.pop(p, None)
                    d = where.get(p)
                    if d:
                        d.pop(w, None)
                        if not d:
                            where.pop(p, None)
                new_w = BPETokenizer._merge(w, best)
                words[new_w] = words.get(new_w, 0) + f
                for p in _get_pairs(new_w):
                    counts[p] += f
                    where.setdefault(p, {})
                    where[p][new_w] = where[p].get(new_w, 0) + f
            counts.pop(best, None)
            where.pop(best, None)
        return cls(vocab, merges)

    # ---- inference ------------------------------------------------------
    def _bpe(self, token: str) -> List[str]:
        word = tuple(token)
        if len(word) < 2:
            return list(word)
        ws_only = all(c in _WS_CHARS for c in word)
        while True:
            pairs = _get_pairs(word)
            if not pairs:
                break
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            # Never grow a whitespace-only run into longer tokens. This keeps
            # runs of spaces as single space tokens, so an all-space loop stays
            # cheap to detect and recover from during generation.
            if ws_only and all(c in self._ws_chars for c in bigram[0]):
                break
            word = self._merge(word, bigram)
            if len(word) == 1:
                break
        return list(word)

    @staticmethod
    def _merge(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        out, i, first, second = [], 0, pair[0], pair[1]
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                out.extend(word[i:])
                break
            out.extend(word[i:j])
            i = j
            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                out.append(first + second)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        ids: List[int] = []
        # split on specials so literal tags map to single ids
        if allow_special:
            pattern = sorted(self.specials, key=len, reverse=True)
            i = 0
            buf = ""
            while i < len(text):
                matched = None
                for s in pattern:
                    if text.startswith(s, i):
                        matched = s
                        break
                if matched:
                    if buf:
                        ids.extend(self._encode_plain(buf))
                        buf = ""
                    ids.append(self.special_to_id[matched])
                    i += len(matched)
                else:
                    buf += text[i]
                    i += 1
            if buf:
                ids.extend(self._encode_plain(buf))
        else:
            ids = self._encode_plain(text)
        return ids

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        cache = self._cache
        for chunk in _pretokenize(text):
            hit = cache.get(chunk)
            if hit is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in chunk.encode("utf-8"))
                hit = [self.encoder.get(p, self.encoder.get("<pad>")) for p in self._bpe(mapped)]
                if len(cache) < 200_000:
                    cache[chunk] = hit
            out.extend(hit)
        return out

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        parts = []
        for i in ids:
            s = self.decoder.get(i, "")
            if skip_special and s in self.special_to_id:
                continue
            parts.append(s)
        text = "".join(parts)
        # Special tokens carry literal ASCII (for example the leading space in
        # " thinking"), so decode them directly. Plain tokens are byte-encoded
        # through the GPT-2 byte<->unicode table.
        out = bytearray()
        for c in text:
            b = UNICODE_TO_BYTE.get(c)
            if b is None:
                out.extend(c.encode("utf-8"))
            else:
                out.append(b)
        return bytes(out).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {"vocab": self.encoder, "merges": self.merges, "specials": self.specials},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], d["merges"], d.get("specials"))
