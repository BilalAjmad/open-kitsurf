"""Pre-norm transformer decoder block (attention + FFN/MoE)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import SwiGLU, MoE
from .rmsnorm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.use_moe = cfg.use_moe
        if cfg.use_moe:
            self.ffn = MoE(
                cfg.hidden_size,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
            )
        else:
            self.ffn = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple] = None,
    ):
        h, new_kv = self.attn(self.attn_norm(x), attention_mask, past_kv)
        x = x + h
        aux = None
        if self.use_moe:
            ffn_out, aux = self.ffn(self.ffn_norm(x))
        else:
            ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, new_kv, aux
