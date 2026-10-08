"""End-to-end demo: build a tiny model, run the agentic loop on a real task.

Run:  PYTHONPATH=src python3 scripts/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from smallm.config import ModelConfig, TextConfig, VisionConfig, param_count
from smallm.model import build_model
from smallm.data import build_tokenizer
from smallm.agent import (
    AgentLoop,
    ScriptedBrain,
    Verifier,
    VirtualComputer,
    default_registry,
    parse_decision,
    validate_english,
)


def banner(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def main():
    cfg = ModelConfig.load(os.path.join(os.path.dirname(__file__), "..", "configs/smallm_600m.json"))
    banner("1. Configuration")
    print(f"600M preset text params (analytic): {param_count(cfg)/1e6:.0f}M")
    print(f"vocab={cfg.text.vocab_size} layers={cfg.text.num_hidden_layers} "
          f"hidden={cfg.text.hidden_size} gqa_kv={cfg.text.num_key_value_heads}")

    banner("2. Multimodal forward pass (tiny stand-in, CPU)")
    tiny = ModelConfig(
        vision=VisionConfig(image_size=32, patch_size=16, hidden_size=64, num_hidden_layers=2,
                            num_attention_heads=4, projector_hidden=128, num_visual_tokens=4),
        text=TextConfig(vocab_size=1024, hidden_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2,
                        intermediate_size=128, max_seq_len=64),
    )
    model = build_model(tiny).eval()
    n_img = model.vision.num_visual_tokens
    ids = torch.randint(8, 1024, (1, 12))
    ids[:, :n_img] = tiny.image_token_id
    out = model(ids, images=torch.randn(1, 3, 32, 32))
    print(f"logits shape = {tuple(out['logits'].shape)}  (vision tokens spliced: {n_img})")

    banner("3. Tokenizer special tokens")
    tok = build_tokenizer(vocab_size=800)
    probe = tok.encode(" thinking plan </think><tool_call>{}</tool_call>")
    print("encoded special probe:", probe)

    banner("4. Parsing a model decision")
    raw = ' thinkingI need to write the file first.</think>' \
          '<tool_call>{"name": "write_file", "args": {"path": "note.txt", "content": "hi"}}</tool_call>'
    print("parsed:", parse_decision(raw))

    banner("5. Strict-English check")
    rep = validate_english("I don't think it's done!")
    print("compliant:", rep.compliant)
    print("violations:", rep.violations)
    print("corrected :", rep.corrected_text)

    banner("6. Full agentic loop on a real task (sandboxed computer)")
    with tempfile.TemporaryDirectory() as workdir:
        computer = VirtualComputer(workdir)
        registry = default_registry(computer)
        brain = ScriptedBrain([
            {"thought": "I write the file in my computer first.",
             "tool": "write_file", "args": {"path": "note.txt", "content": "hello"}},
            {"thought": "Now I read the file to check the result.",
             "tool": "read_file", "args": {"path": "note.txt"}},
            {"thought": "The content matches. I check it and answer.",
             "answer": "I wrote note.txt. The file contains hello."},
        ])
        loop = AgentLoop(brain, registry, Verifier(required_keywords=["hello"]))
        result = loop.run("Create note.txt with hello. Then check the file.")
        for s in result.steps:
            print(f"[{s.kind:12}] {s.content[:90]}")
        print(f"\nPASSED={result.passed} iterations={result.iterations}")
        print("final answer:", result.answer)


if __name__ == "__main__":
    main()
