"""Model soup: merge several fine-tunes of the same architecture into one model.

Two modes:
  - uniform soup: simple element-wise average of weights. Robust, cheap.
  - greedy soup: start from the best model and add each candidate only when it
    improves validation loss. Needs a loss function, so it costs evaluations.

Reference idea: Wortsman et al., "Model Soups" (2022).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch


def _flatten_state(sd: Dict[str, torch.Tensor]):
    keys = sorted(sd.keys())
    return keys


def uniform_soup(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("no state dicts given")
    keys = state_dicts[0].keys()
    out: Dict[str, torch.Tensor] = {}
    n = float(len(state_dicts))
    for k in keys:
        acc = None
        for sd in state_dicts:
            v = sd[k]
            if not torch.is_floating_point(v):
                # integer buffers (e.g. step counters): take the first
                acc = v.clone() if acc is None else acc
                continue
            acc = v.float() if acc is None else acc + v.float()
        out[k] = acc.div(n).to(state_dicts[0][k].dtype) if torch.is_floating_point(acc) else acc
    return out


def greedy_soup(
    state_dicts: List[Dict[str, torch.Tensor]],
    eval_loss: Callable[[Dict[str, torch.Tensor]], float],
) -> tuple[Dict[str, torch.Tensor], List[int]]:
    """Greedily add models that lower validation loss."""
    scores = [eval_loss(sd) for sd in state_dicts]
    order = sorted(range(len(state_dicts)), key=lambda i: scores[i])
    best_idx = order[0]
    soup = {k: v.clone() for k, v in state_dicts[best_idx].items()}
    best_loss = scores[best_idx]
    chosen = [best_idx]
    count = 1

    for i in order[1:]:
        cand = {k: (soup[k].float() * count + state_dicts[i][k].float()) / (count + 1)
                for k in soup}
        cand = {k: (v.to(soup[k].dtype) if torch.is_floating_point(v) else v)
                for k, v in cand.items()}
        loss = eval_loss(cand)
        if loss < best_loss:
            soup = cand
            best_loss = loss
            count += 1
            chosen.append(i)
    return soup, chosen


def save_soup(soup: Dict[str, torch.Tensor], config: dict, path: str) -> None:
    torch.save({"model": soup, "config": config}, path)
