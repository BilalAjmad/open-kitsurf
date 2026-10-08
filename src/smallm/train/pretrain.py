"""Minimal CPU/GPU training loop for causal LM + optional image conditioning."""
from __future__ import annotations

from typing import Optional

import torch


def make_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.1):
    # no weight decay on norms and biases
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def train_epoch(model, loader, optimizer, device: str = "cpu", max_steps: Optional[int] = None) -> dict:
    model.train()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        images = batch.get("images")
        if images is not None:
            images = images.to(device)
        out = model(input_ids=input_ids, images=images, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
        n += 1
    return {"loss": total / max(n, 1), "steps": n}
