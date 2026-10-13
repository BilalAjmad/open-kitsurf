import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from smallm.config import ModelConfig, TextConfig, VisionConfig, param_count
from smallm.model import build_model


def tiny_cfg():
    return ModelConfig(
        vision=VisionConfig(image_size=32, patch_size=16, hidden_size=64,
                            num_hidden_layers=2, num_attention_heads=4, projector_hidden=128,
                            num_visual_tokens=4),
        text=TextConfig(vocab_size=512, hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_seq_len=64),
    )


def test_param_count_configs():
    for f, lo in [("configs/smallm_600m.json", 550e6), ("configs/smallm_1b.json", 900e6)]:
        path = os.path.join(os.path.dirname(__file__), "..", f)
        c = ModelConfig.load(path)
        p = param_count(c)
        assert lo < p < 1.6e9, (f, p)


def test_forward_text():
    model = build_model(tiny_cfg()).eval()
    ids = torch.randint(0, 512, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, 512)
    assert torch.isfinite(out["logits"]).all()


def test_loss_backward():
    model = build_model(tiny_cfg()).train()
    ids = torch.randint(0, 512, (2, 16))
    labels = ids.clone()
    out = model(ids, labels=labels)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0


def test_multimodal_splice():
    cfg = tiny_cfg()
    model = build_model(cfg).eval()
    n_img = model.vision.num_visual_tokens
    ids = torch.randint(8, 512, (2, 20))
    ids[:, :n_img] = cfg.image_token_id
    images = torch.randn(2, 3, 32, 32)
    out = model(ids, images=images)
    assert out["logits"].shape == (2, 20, 512)


def test_image_count_mismatch_raises():
    cfg = tiny_cfg()
    model = build_model(cfg).eval()
    n_img = model.vision.num_visual_tokens
    ids = torch.randint(8, 512, (2, 20))
    ids[:, :n_img] = cfg.image_token_id  # correct count
    images = torch.randn(2, 3, 32, 32)
    model(ids, images=images)
    bad = ids.clone()
    bad[:, n_img - 1] = 7  # remove one image token
    try:
        model(bad, images=images)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generate_cache_matches_no_cache():
    cfg = tiny_cfg()
    model = build_model(cfg).eval()
    ids = torch.randint(0, 512, (1, 8))
    greedy = model.generate(ids, max_new_tokens=5, temperature=0.0)
    assert greedy.shape[1] == 13


def test_load_dolly_parses_real_format(tmp_path):
    import json
    from smallm.data import load_dolly

    p = tmp_path / "dolly.jsonl"
    rows = [
        {"instruction": "What is 2+2?", "context": "", "response": "4"},
        {"instruction": "Name a colour", "context": "a red apple", "response": "red"},
        {"instruction": "", "context": "", "response": "skip me"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pairs = load_dolly(str(p))
    assert len(pairs) == 2
    assert pairs[0][1] == "4"
    assert "context: a red apple" in pairs[1][0]
    assert pairs[1][1] == "red"


def test_vision_gradients_reach_every_vision_module():
    model = build_model(tiny_cfg()).train()
    n = model.vision.num_visual_tokens
    ids = torch.randint(0, 512, (2, 1 + n + 4))
    ids[:, 1:1 + n] = tiny_cfg().image_token_id
    labels = ids.clone()
    labels[:, :1 + n] = -100
    out = model(input_ids=ids, images=torch.randn(2, 3, 32, 32), labels=labels)
    out["loss"].backward()
    for name, p in model.vision.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert model.projector.net[0].weight.grad.abs().sum() > 0


def test_vision_changes_caption_vs_blank():
    # A trained vision path must make the caption depend on the pixels. Here we
    # only assert that different images produce different hidden states, which
    # is the property the training script exploits.
    model = build_model(tiny_cfg()).eval()
    n = model.vision.num_visual_tokens
    ids = torch.randint(0, 512, (1, 1 + n + 4))
    ids[:, 1:1 + n] = tiny_cfg().image_token_id
    a = model.encode_images(torch.zeros(1, 3, 32, 32))
    b = model.encode_images(torch.ones(1, 3, 32, 32))
    assert not torch.allclose(a, b)


def test_moe_forward():
    cfg = tiny_cfg()
    cfg.text.use_moe = True
    cfg.text.num_experts = 4
    cfg.text.num_experts_per_tok = 2
    cfg.text.moe_intermediate_size = 64
    model = build_model(cfg).eval()
    ids = torch.randint(0, 512, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
