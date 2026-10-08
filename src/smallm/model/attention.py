"""Causal self-attention with grouped-query attention and RoPE."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = cfg.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple] = None,
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0 if past_kv is None else past_kv[0].shape[-2]
        cos, sin = self.rope.forward(offset + T)
        cos, sin = cos[offset:], sin[offset:]
        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v)

        # expand kv heads to query heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.shape[-2]
        causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(
            diagonal=S - T
        )
        if attention_mask is not None:
            # attention_mask: (B, S) with 1 for keep, 0 for pad
            am = attention_mask[:, None, None, :].bool() & causal[None, None]
        else:
            am = causal[None, None]

        dropout_p = 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv
