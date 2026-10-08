"""Distillation trainer: a large teacher supervises the small student.

Two signals are combined:
  1. hard-label cross-entropy on the teacher's reasoning traces
  2. KL divergence between student and teacher next-token distributions
     (soft labels) at temperature T
The teacher only needs to expose `logits_for(input_ids)`. It can be a local
model, an API-cached logit dump, or a frozen copy of a bigger SmallM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillConfig:
    temperature: float = 2.0
    alpha_soft: float = 0.5   # weight on KL(teacher || student)
    alpha_hard: float = 0.5   # weight on cross-entropy to hard targets
    use_vision_loss: bool = False


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: Optional[torch.Tensor] = None,
    cfg: Optional[DistillConfig] = None,
):
    cfg = cfg or DistillConfig()
    T = cfg.temperature
    V = student_logits.size(-1)
    s = student_logits.view(-1, V)
    t = teacher_logits.view(-1, V)

    soft = F.kl_div(
        F.log_softmax(s / T, dim=-1),
        F.softmax(t / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

    total = cfg.alpha_soft * soft
    if hard_labels is not None and cfg.alpha_hard > 0:
        # cross-entropy over the teacher's argmax when no hard labels exist
        ce = F.cross_entropy(s, hard_labels.view(-1), ignore_index=-100)
        total = total + cfg.alpha_hard * ce
    return total, soft.detach()


class DistillTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher_logits_fn: Callable[[torch.Tensor], torch.Tensor],
        cfg: Optional[DistillConfig] = None,
    ) -> None:
        self.student = student
        self.teacher_logits_fn = teacher_logits_fn
        self.cfg = cfg or DistillConfig()

    def step(self, batch: dict, optimizer: torch.optim.Optimizer) -> dict:
        input_ids = batch["input_ids"]
        images = batch.get("images")
        labels = batch.get("labels")

        out = self.student(
            input_ids=input_ids,
            images=images,
            labels=labels,  # gives the hard CE term when labels exist
        )
        student_logits = out["logits"]

        with torch.no_grad():
            teacher_logits = self.teacher_logits_fn(input_ids)

        loss, soft = distillation_loss(
            student_logits, teacher_logits, hard_labels=labels, cfg=self.cfg
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "kl": float(soft),
            "aux": float(out.get("aux_loss", torch.tensor(0.0))),
        }
