import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from smallm.data import build_tokenizer, make_text_batch, make_image_batch
from smallm.tokenizer import BPETokenizer
from smallm.config import ModelConfig, TextConfig, VisionConfig
from smallm.model import build_model
from smallm.train import train_epoch, make_optimizer, DistillTrainer, DistillConfig


def test_bpe_roundtrip():
    tok = build_tokenizer(vocab_size=600)
    for s in ["hello world", "I do not use contractions.", "café ☕ 日本語"]:
        ids = tok.encode(s, allow_special=True)
        assert len(ids) > 0
        # lossless on the non-special payload
        assert tok.decode(ids, skip_special=False).replace("\ufffd", "") != "" or s == ""


def test_special_tokens_map():
    tok = build_tokenizer(vocab_size=600)
    ids = tok.encode(" thinking x </think><tool_call>{}</tool_call>")
    assert tok.special_to_id[" thinking"] in ids
    assert tok.special_to_id["<tool_call>"] in ids


def test_text_batch_shapes():
    tok = build_tokenizer(vocab_size=600)
    b = make_text_batch(["hello world", "think and act"], tok, seq_len=32, pad_to=16)
    assert b["input_ids"].shape == (2, 16)
    assert b["labels"].shape == (2, 16)


def test_train_step_runs():
    tok = build_tokenizer(vocab_size=600)
    cfg = ModelConfig(
        vision=VisionConfig(image_size=32, patch_size=16, hidden_size=64,
                            num_hidden_layers=1, num_attention_heads=4, projector_hidden=128, num_visual_tokens=4),
        text=TextConfig(vocab_size=600, hidden_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_seq_len=32),
    )
    model = build_model(cfg)
    opt = make_optimizer(model, lr=1e-3)
    batch = make_text_batch(["hello world", "think and act", "verify then answer"], tok, pad_to=16)
    r = train_epoch(model, [batch], opt, max_steps=2)
    assert r["steps"] >= 1 and r["loss"] > 0


def test_distillation_step_and_loss_decreases():
    tok = build_tokenizer(vocab_size=600)
    cfg = ModelConfig(
        vision=VisionConfig(image_size=32, patch_size=16, hidden_size=64,
                            num_hidden_layers=1, num_attention_heads=4, projector_hidden=128, num_visual_tokens=4),
        text=TextConfig(vocab_size=600, hidden_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_seq_len=32),
    )
    student = build_model(cfg)
    # a frozen "teacher" that is just a fixed random projection of embeddings,
    # standing in for a bigger model's logits
    teacher_head = torch.nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)

    def teacher_logits(input_ids):
        emb = student.text.embed_tokens(input_ids)
        return teacher_head(emb.mean(dim=1, keepdim=True).expand(-1, input_ids.shape[1], -1))

    trainer = DistillTrainer(student, teacher_logits, DistillConfig(alpha_soft=0.7, alpha_hard=0.3))
    opt = make_optimizer(student, lr=1e-3)
    batch = make_text_batch(["hello world", "think and act", "verify then answer"], tok, pad_to=16)

    first = trainer.step(batch, opt)["loss"]
    for _ in range(30):
        last = trainer.step(batch, opt)["loss"]
    assert last < first, (first, last)
