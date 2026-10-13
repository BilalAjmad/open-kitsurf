"""Top-level multimodal agentic language model: SmallM."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .ageis import AgeisModel
from .nova import NovaModel
from .text_model import TextModel
from .vision import Projector, VisionEncoder


def build_text_backbone(cfg):
    """Pick the text backbone from `cfg.arch`."""
    arch = getattr(cfg, "arch", "transformer")
    if arch == "ageis":
        return AgeisModel(cfg)
    if arch == "nova":
        return NovaModel(cfg)
    if arch == "transformer":
        return TextModel(cfg)
    raise ValueError(f"unknown text arch: {arch!r}")


class SmallM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision = VisionEncoder(cfg.vision)
        self.projector = Projector(cfg.vision, cfg.text.hidden_size)
        self.text = build_text_backbone(cfg.text)
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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        image_embeds = self.encode_images(images) if images is not None else None
        cur = input_ids
        # Prime the cache with the prompt (image embeds only on first step).
        out = self.text(cur, image_embeds=image_embeds, use_cache=True, past_key_values=None)
        past = out["past_key_values"]
        next_tok = out["logits"][:, -1, :]
        for _ in range(max_new_tokens):
            logits = next_tok.clone()
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(cur.shape[0]):
                    seen = set(cur[b].tolist())
                    for t in seen:
                        logits[b, t] = logits[b, t] / repetition_penalty if logits[b, t] > 0 \
                            else logits[b, t] * repetition_penalty
            if no_repeat_ngram_size and cur.shape[1] >= no_repeat_ngram_size:
                n = no_repeat_ngram_size
                for b in range(cur.shape[0]):
                    seq = cur[b].tolist()
                    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                    banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                              if tuple(seq[i:i + n - 1]) == prefix}
                    for t in banned:
                        logits[b, t] = -float("inf")
            if temperature <= 0:
                tok = logits.argmax(-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(scaled, top_k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < kth, -float("inf"))
                probs = torch.softmax(scaled, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)
            cur = torch.cat([cur, tok], dim=1)
            if eos_token_id is not None and (tok == eos_token_id).all():
                break
            out = self.text(tok, use_cache=True, past_key_values=past)
            past = out["past_key_values"]
            next_tok = out["logits"][:, -1, :]
        return cur


def build_model(cfg: ModelConfig) -> SmallM:
    return SmallM(cfg)
