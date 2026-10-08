"""Decoder-only text backbone with multimodal input splicing."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .block import DecoderBlock
from .rmsnorm import RMSNorm


class TextModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_token_id = 3  # overridden by SmallM from the top-level config
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def embed_only(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        image_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        if image_embeds is not None:
            itid = self.image_token_id if image_token_id is None else image_token_id
            b, t, _ = x.shape
            mask = input_ids == itid  # (B, T)
            counts = mask.sum(dim=1)
            n_img = image_embeds.shape[1]
            # Each sample must carry exactly n_img image tokens in sequence order.
            if not torch.all(counts == n_img):
                raise ValueError(
                    f"image token count per sample {counts.tolist()} != image embed count {n_img}"
                )
            # nonzero is row-major: already (batch, position) ordered, matching
            # image_embeds flattened as (B, n_img, D) -> (B*n_img, D).
            flat_positions = mask.nonzero(as_tuple=False)  # (B*n_img, 2)
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
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, kv, aux = layer(x, attention_mask, past)
            if use_cache:
                new_kvs.append(kv)
            if aux is not None:
                aux_total = aux_total + aux
        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "hidden": x, "aux_loss": aux_total, "past_key_values": new_kvs}
