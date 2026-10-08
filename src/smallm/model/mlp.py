"""SwiGLU feed-forward and an optional sparse mixture-of-experts variant."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Top-k sparse mixture of SwiGLU experts with a learned router.

    Keeps active compute near a dense model of `moe_intermediate_size`
    while the total parameter count grows with the number of experts.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        expert_intermediate: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_size, expert_intermediate) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat)
        weights, idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(flat)
        # load-balancing auxiliary loss (Switch-transformer style)
        aux = flat.new_zeros(())
        for e in range(self.num_experts):
            mask = (idx == e).any(dim=-1)
            if not mask.any():
                continue
            expert_out = self.experts[e](flat[mask])
            # weight each token by its gate value for that expert
            w = weights[mask][(idx[mask] == e)]
            out[mask] += expert_out * w.unsqueeze(-1)
            # aux: fraction routed * mean router prob
            frac = mask.float().mean()
            prob = F.softmax(logits, dim=-1)[:, e].mean()
            aux = aux + frac * prob
        aux = aux * self.num_experts
        return out.view(B, T, D), aux
