"""SmallM-Nova: a recurrent-depth (looped) transformer backbone.

A genuinely different architecture from the plain stack in `text_model.py`:

  * Recurrent depth ("loop engineering"): a small stack of unique blocks is
    applied `num_recurrent_steps` times. A learned per-step embedding is added
    to the hidden state on each pass, so the loop is not a no-op. Depth grows
    without new parameters.
  * Parallel residual (GPT-J / Gemma style): attention and feed-forward read the
    *same* normalized input and both write to the residual stream.
  * QK-norm: query and key are RMS-normalized per head, which stabilizes
    training at high learning rates (Chameleon / Gemma-2 style).
  * Layer scale: each residual branch is scaled by a learned per-channel gain,
    which lets deep/looped stacks start near identity.

`NovaModel` keeps the exact forward contract of `TextModel` (same argument names
and the same dict keys), so it is a drop-in text backbone for `SmallM`.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import SwiGLU, MoE
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


class NovaAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = cfg.hidden_size // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.qk_norm = bool(getattr(cfg, "qk_norm", False))

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x, attention_mask=None, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        offset = 0 if past_kv is None else past_kv[0].shape[-2]
        cos, sin = self.rope.forward(offset + T)
        cos, sin = cos[offset:], sin[offset:]
        q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)
            v = torch.cat([past_kv[1], v], dim=-2)
        new_kv = (k, v)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.shape[-2]
        causal = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        if attention_mask is not None:
            am = attention_mask[:, None, None, :].bool() & causal[None, None]
        else:
            am = causal[None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


class NovaBlock(nn.Module):
    """Pre-norm block with a parallel attention + FFN residual."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.parallel = bool(getattr(cfg, "parallel_residual", True))
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = NovaAttention(cfg)
        self.use_moe = cfg.use_moe
        if cfg.use_moe:
            self.ffn = MoE(cfg.hidden_size, cfg.num_experts,
                           cfg.num_experts_per_tok, cfg.moe_intermediate_size)
        else:
            self.ffn = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        if not self.parallel:
            self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        ls = float(getattr(cfg, "layer_scale_init", 0.0))
        if ls > 0:
            self.attn_scale = nn.Parameter(torch.full((cfg.hidden_size,), ls))
            self.ffn_scale = nn.Parameter(torch.full((cfg.hidden_size,), ls))
        else:
            self.attn_scale = None
            self.ffn_scale = None

    def forward(self, x, attention_mask=None, past_kv=None):
        h = self.attn_norm(x)
        attn_out, new_kv = self.attn(h, attention_mask, past_kv)
        if self.attn_scale is not None:
            attn_out = attn_out * self.attn_scale

        aux = None
        if self.parallel:
            ffn_in = h  # same normalized input as attention
        else:
            ffn_in = self.ffn_norm(x + attn_out)
        if self.use_moe:
            ffn_out, aux = self.ffn(ffn_in)
        else:
            ffn_out = self.ffn(ffn_in)
        if self.ffn_scale is not None:
            ffn_out = ffn_out * self.ffn_scale

        x = x + attn_out + ffn_out
        return x, new_kv, aux


class NovaModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_token_id = 3
        self.steps = max(1, int(getattr(cfg, "num_recurrent_steps", 1)))
        self.num_layers = cfg.num_hidden_layers
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([NovaBlock(cfg) for _ in range(self.num_layers)])
        # one learned embedding per recurrence pass ("loop engineering")
        self.step_embed = nn.Parameter(torch.zeros(self.steps, cfg.hidden_size))
        nn.init.normal_(self.step_embed, std=0.02)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def embed_only(self, input_ids, image_embeds=None, image_token_id=None):
        x = self.embed_tokens(input_ids)
        if image_embeds is not None:
            itid = self.image_token_id if image_token_id is None else image_token_id
            mask = input_ids == itid
            counts = mask.sum(dim=1)
            n_img = image_embeds.shape[1]
            if not torch.all(counts == n_img):
                raise ValueError(
                    f"image token count per sample {counts.tolist()} != image embed count {n_img}"
                )
            flat_positions = mask.nonzero(as_tuple=False)
            x = x.clone()
            x[flat_positions[:, 0], flat_positions[:, 1]] = image_embeds.reshape(
                -1, image_embeds.shape[-1]
            ).to(x.dtype)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[tuple]] = None,
    ):
        x = self.embed_only(input_ids, image_embeds)
        new_kvs = [] if use_cache else None
        aux_total = x.new_zeros(())
        for s in range(self.steps):
            x = x + self.step_embed[s].to(x.dtype)
            for i, layer in enumerate(self.layers):
                idx = s * self.num_layers + i
                past = None if past_key_values is None else past_key_values[idx]
                x, kv, aux = layer(x, attention_mask, past)
                if use_cache:
                    new_kvs.append(kv)
                if aux is not None:
                    aux_total = aux_total + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "hidden": x, "aux_loss": aux_total,
                "past_key_values": new_kvs}
