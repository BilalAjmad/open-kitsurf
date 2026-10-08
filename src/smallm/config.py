"""Model and training configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    mlp_ratio: float = 4.0
    projector_hidden: int = 1024
    num_visual_tokens: int = 64


@dataclass
class TextConfig:
    vocab_size: int = 49152
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # grouped-query attention
    intermediate_size: int = 2816  # SwiGLU
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    # Optional mixture-of-experts (off by default -> dense model)
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 768


@dataclass
class ModelConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    image_token_id: int = 3  # placeholder token id replaced by vision embeddings
    # Special tokens used by the agentic / thinking tokenizer
    think_open_id: int = 4
    think_close_id: int = 5
    verify_open_id: int = 6
    verify_close_id: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            vision=VisionConfig(**d["vision"]),
            text=TextConfig(**d["text"]),
            **{k: v for k, v in d.items() if k not in ("vision", "text")},
        )

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def param_count(cfg: ModelConfig) -> int:
    """Rough analytic parameter count (no forward pass needed)."""
    t = cfg.text
    h = t.hidden_size
    # attention: q proj + kv proj (GQA) + o proj
    kv_dim = h // t.num_attention_heads * t.num_key_value_heads
    attn = h * h + 2 * h * kv_dim + h * h
    # SwiGLU mlp: 3 matrices (or MoE: router + num_experts experts)
    if t.use_moe:
        mlp = h * t.num_experts + t.num_experts * 3 * h * t.moe_intermediate_size
    else:
        mlp = 3 * h * t.intermediate_size
    per_layer = attn + mlp + 2 * h  # two RMSNorm weights
    text = per_layer * t.num_hidden_layers
    text += t.vocab_size * h  # embeddings (tied lm_head)
    # vision tower (ViT-like)
    v = cfg.vision
    v_attn = 4 * v.hidden_size * v.hidden_size
    v_mlp = int(v.mlp_ratio) * 2 * v.hidden_size * v.hidden_size
    vis = (v_attn + v_mlp) * v.num_hidden_layers
    vis += v.in_channels * v.patch_size * v.patch_size * v.hidden_size
    # projector (2-layer MLP into text space)
    proj = v.hidden_size * v.projector_hidden + v.projector_hidden * h
    return text + vis + proj
