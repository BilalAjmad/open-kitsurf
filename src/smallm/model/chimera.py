"""SmallM-Chimera: a hybrid selective-state + grouped-query attention model.

Chimera is a third, independent architecture in this repo, built around a
hybrid of two sequence-mixing primitives that 2026 frontier models (Mamba,
Jamba, Samba) converged on:

  * **Selective state-space mixer (SSM).** A diagonal, input-dependent linear
    recurrence ``h_t = a_t * h_{t-1} + (1 - a_t) * x_t`` where ``a_t`` is a
    *learned, data-dependent* per-channel decay. This gives linear-time mixing
    with a forget gate the model controls per token and per channel. Unlike a
    convolution it can hold or drop state based on content.
  * **Grouped-query attention (GQA)** with RoPE, for precise content-addressed
    recall over long range, which a fixed-width state cannot provide.

The two are *interleaved* by depth (most layers attention, every fourth an SSM)
so the stack gets cheap long-range memory plus exact recall. On top:

  * **Mixture-of-experts FFN.** A sparse router sends each token to the top-k
    experts, so total capacity grows without a matching compute cost.
  * **Sandwich RMSNorm** after every residual add, for stable deep training.
  * **Tied embeddings**, GQA, QK-norm, RoPE.

This is distinct from the plain stack (`text_model.py`), Nova (`nova.py`, a
recurrent-depth loop) and Ageis (`ageis.py`, a gated loop with sandwich norms):
Chimera's novelty is the SSM/attention hybrid and the sparse FFN.

`ChimeraModel` keeps the exact forward contract of the other backbones (same
argument names, same dict keys), so it is a drop-in text backbone.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import SwiGLU, MoE
from .rmsnorm import RMSNorm
from .rope import RotaryEmbedding, apply_rope


class SelectiveSSM(nn.Module):
    """Diagonal selective state-space mixer with a learned per-channel decay."""

    def __init__(self, cfg) -> None:
        super().__init__()
        h = cfg.hidden_size
        self.in_proj = nn.Linear(h, 2 * h, bias=False)
        self.a_proj = nn.Linear(h, h, bias=True)
        self.out_proj = nn.Linear(h, h, bias=False)
        # bias starts slightly positive so initial decay is near 1 (long memory)
        nn.init.constant_(self.a_proj.bias, 0.5)
        self.norm = RMSNorm(h, cfg.rms_norm_eps)

    def forward(self, x, past_state=None):
        g, v = self.in_proj(x).chunk(2, dim=-1)
        g = F.silu(g)
        a = torch.sigmoid(self.a_proj(x))          # (B,T,D) in (0,1)
        inp = g * v
        B, T, D = inp.shape
        h = inp.new_zeros(B, D) if past_state is None else past_state[0]
        outs = []
        for t in range(T):
            at = a[:, t]
            h = at * h + (1.0 - at) * inp[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1)
        return self.out_proj(self.norm(y)), (h,)


class ChimeraAttention(nn.Module):
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
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x, attention_mask=None, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

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


class ChimeraBlock(nn.Module):
    """Sandwich-norm block whose mixer is attention or SSM depending on depth."""

    def __init__(self, cfg, use_ssm: bool) -> None:
        super().__init__()
        self.use_ssm = use_ssm
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if use_ssm:
            self.mixer = SelectiveSSM(cfg)
        else:
            self.mixer = ChimeraAttention(cfg)
        self.post1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.use_moe = cfg.use_moe
        if cfg.use_moe:
            self.ffn = MoE(cfg.hidden_size, cfg.num_experts,
                           cfg.num_experts_per_tok, cfg.moe_intermediate_size)
        else:
            self.ffn = SwiGLU(cfg.hidden_size, cfg.intermediate_size)
        self.post2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, attention_mask=None, past_kv=None):
        h = self.norm1(x)
        if self.use_ssm:
            mixed, new_kv = self.mixer(h, past_kv)
        else:
            mixed, new_kv = self.mixer(h, attention_mask, past_kv)
        x = x + self.post1(mixed)

        aux = None
        h2 = self.norm2(x)
        if self.use_moe:
            ffn_out, aux = self.ffn(h2)
        else:
            ffn_out = self.ffn(h2)
        x = x + self.post2(ffn_out)
        return x, new_kv, aux


class ChimeraModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_token_id = 3
        # every `ssm_every`-th layer is an SSM; the rest are attention
        self.ssm_every = max(1, int(getattr(cfg, "ssm_every", 4)))
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            ChimeraBlock(cfg, use_ssm=(i % self.ssm_every == self.ssm_every - 1))
            for i in range(cfg.num_hidden_layers)
        ])
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
            flat = mask.nonzero(as_tuple=False)
            x = x.clone()
            x[flat[:, 0], flat[:, 1]] = image_embeds.reshape(
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
        cache_idx = 0
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[cache_idx]
            x, kv, aux = layer(x, attention_mask, past)
            if use_cache:
                new_kvs.append(kv)
            cache_idx += 1
            if aux is not None:
                aux_total = aux_total + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "hidden": x, "aux_loss": aux_total,
                "past_key_values": new_kvs}
