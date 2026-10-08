"""Top-level multimodal agentic language model: SmallM."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .text_model import TextModel
from .vision import Projector, VisionEncoder


class SmallM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg.vision)
        self.projector = Projector(cfg.vision, cfg.text.hidden_size)
        self.text = TextModel(cfg.text)
        self.text.image_token_id = cfg.image_token_id

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision(images))

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values=None,
    ):
        image_embeds = self.encode_images(images) if images is not None else None
        out = self.text(
            input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        loss = None
        if labels is not None:
            logits = out["logits"][:, :-1, :].contiguous()
            tgt = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100
            )
            if self.cfg.text.use_moe:
                loss = loss + 0.01 * out["aux_loss"]
        out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        image_embeds = self.encode_images(images) if images is not None else None
        past = None
        cur = input_ids
        # Prime the cache with the prompt (image embeds only on first step).
        out = self.text(cur, image_embeds=image_embeds, use_cache=True, past_key_values=None)
        past = out["past_key_values"]
        next_tok = out["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                tok = next_tok.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(next_tok / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, tok], dim=1)
            if eos_token_id is not None and (tok == eos_token_id).all():
                break
            out = self.text(
                tok, use_cache=True, past_key_values=past
            )
            past = out["past_key_values"]
            next_tok = out["logits"][:, -1, :]
        return cur


def build_model(cfg: ModelConfig) -> SmallM:
    return SmallM(cfg)
