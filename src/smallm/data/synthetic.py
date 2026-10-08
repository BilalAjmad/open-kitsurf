"""Synthetic multimodal + agentic data for tests and smoke training.

Real training replaces this with a tokenized corpus and an image folder.
The important part is that the *format* matches what the agent expects:
thinking traces, tool calls, strict-English answers, and image conditioning.
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch

from ..tokenizer.bpe import BPETokenizer

AGENTIC_SAMPLES = [
    "Write a file. First I check the sandbox. <tool_call>{\"name\": \"write_file\", "
    "\"args\": {\"path\": \"a.txt\", \"content\": \"hello\"}}</tool_call>",
    "Now I read it back to verify. <tool_call>{\"name\": \"read_file\", "
    "\"args\": {\"path\": \"a.txt\"}}</tool_call>",
    "The read result matches. I answer. The file contains hello.",
    "List the files. <tool_call>{\"name\": \"list_dir\", \"args\": {\"path\": \".\"}}</tool_call>",
    "I verified the listing. The file a.txt exists.",
]

STRICT_SAMPLES = [
    "I do not use contractions. This sentence is short. The answer is correct.",
    "Think first. Then act. Then verify. This is the agent loop.",
]


def token_corpus() -> List[str]:
    return AGENTIC_SAMPLES + STRICT_SAMPLES


def build_tokenizer(vocab_size: int = 1024) -> BPETokenizer:
    return BPETokenizer.train(token_corpus(), vocab_size=vocab_size, min_freq=1)


def make_text_batch(
    texts: List[str],
    tokenizer: BPETokenizer,
    seq_len: int = 64,
    pad_to: int = 32,
) -> Dict[str, torch.Tensor]:
    rows, labels = [], []
    for t in texts:
        ids = tokenizer.encode(t, allow_special=True)[:seq_len]
        lab = list(ids)
        while len(ids) < pad_to:
            ids.append(tokenizer.pad_id)
            lab.append(-100)
        rows.append(ids[:pad_to])
        labels.append(lab[:pad_to])
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_image_batch(
    texts: List[str],
    tokenizer: BPETokenizer,
    image_size: int = 32,
    num_visual_tokens: int = 4,
    pad_to: int = 32,
) -> Dict[str, torch.Tensor]:
    """Text with `num_visual_tokens` <image> tokens, plus random images."""
    rows, labels, imgs = [], [], []
    for t in texts:
        prefix = [tokenizer.image_token_id] * num_visual_tokens
        ids = prefix + tokenizer.encode(t, allow_special=True)
        ids = ids[:pad_to]
        lab = list(ids)
        lab[:num_visual_tokens] = [-100] * min(num_visual_tokens, len(lab))
        while len(ids) < pad_to:
            ids.append(tokenizer.pad_id)
            lab.append(-100)
        rows.append(ids)
        labels.append(lab)
        imgs.append(torch.randn(3, image_size, image_size))
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "images": torch.stack(imgs),
    }
