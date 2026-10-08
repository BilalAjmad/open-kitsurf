# Repository notes for agents

## What this is
SmallM: a from-scratch small (~600M/1B) multimodal, agentic, self-verifying
language model. Model + tokenizer + agent + distillation code. No pretrained
weights are shipped.

## Environment
- Dev sandbox: 4 CPU cores, ~15 GB RAM, no GPU. Torch is CPU-only.
- Install: `pip install -r requirements.txt pytest`
- Test: `PYTHONPATH=src python3 -m pytest tests/ -q` (all tests must pass on CPU)
- Demo: `PYTHONPATH=src python3 scripts/demo.py`

## Layout
- `src/smallm/model/` transformer, vision, multimodal top-level
- `src/smallm/agent/` loop, tools, computer sandbox, verifier, strict English
- `src/smallm/tokenizer/` byte-level BPE with agentic special tokens
- `src/smallm/train/` training + distillation
- `src/smallm/data/` synthetic data (tests only)
- `configs/` 600M dense and 1B MoE presets

## Conventions
- Keep tests CPU-runnable and fast. Do not instantiate the full 600M/1B model
  in tests; use `tiny_cfg()` style presets.
- Special token ids are fixed in the tokenizer: `<image>`=3, ` thinking`=4,
   `</think>`=5, `<verify>`=6, `</verify>`=7, then tool tokens 8-13.
- Be honest in docs: never claim trained weights or frontier-beating results.
