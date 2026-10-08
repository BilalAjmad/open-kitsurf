"""A compact ViT-style vision encoder plus an MLP projector into text space.

Native-resolution patch embedding keeps the design simple and CPU-testable.
Swap in a SigLIP/CLIP tower for production distillation without touching the
projector contract (`(B, N_patches, vision_hidden) -> (B, N_patches, text_hidden)`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)  # (B, H, h, w)
        return x.flatten(2).transpose(1, 2)  # (B, N, H)


class ViTBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_size, cfg.num_attention_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        inter = int(cfg.hidden_size * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, inter),
            nn.GELU(),
            nn.Linear(inter, cfg.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, cfg.hidden_size))
        # Perceiver-style resampler to a fixed number of visual tokens.
        self.num_visual_tokens = cfg.num_visual_tokens
        self.latents = nn.Parameter(torch.randn(1, self.num_visual_tokens, cfg.hidden_size) * 0.02)
        self.resample_attn = nn.MultiheadAttention(cfg.hidden_size, cfg.num_attention_heads, batch_first=True)
        self.resample_norm = nn.LayerNorm(cfg.hidden_size)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        latents = self.latents.expand(b, -1, -1)
        q = self.resample_norm(latents)
        out, _ = self.resample_attn(q, x, x, need_weights=False)
        return out  # (B, num_visual_tokens, vision_hidden)


class Projector(nn.Module):
    """Two-layer MLP mapping visual tokens into the text embedding space."""

    def __init__(self, cfg, text_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.projector_hidden),
            nn.GELU(),
            nn.Linear(cfg.projector_hidden, text_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
