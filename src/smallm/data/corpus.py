"""Memory-mapped token corpus and random-crop batch sampler."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator

import numpy as np
import torch


class TokenCorpus:
    def __init__(self, bin_path: str, meta_path: str | None = None) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n = int(self.data.shape[0])
        self.meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor]:
        # A sample is seq_len inputs plus the next token as the last label.
        max_start = self.n - seq_len - 2
        idx = torch.randint(0, max_start, (batch_size,), generator=generator)
        rows = np.empty((batch_size, seq_len + 1), dtype=np.int64)
        for b, s in enumerate(idx.tolist()):
            rows[b] = self.data[s : s + seq_len + 1]
        t = torch.from_numpy(rows).long()
        return {"input_ids": t[:, :-1], "labels": t[:, 1:]}

    def iter_batches(
        self,
        batch_size: int,
        seq_len: int,
        steps: int,
        seed: int = 0,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        g = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            yield self.sample_batch(batch_size, seq_len, g)
