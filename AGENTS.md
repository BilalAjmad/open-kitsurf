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

## Agentic training notes (learned on CPU)
- SFT traces run 190–340 tokens. Train with `--seq-len 384`, not 256, or
  long traces get truncated.
- Use response-only loss (`sft_agentic.py` default). Full-sequence loss
  (`--train-on-prompt`) overfits the prompt and hurts held-out copying.
- Decode the agent with greedy + a *gentle* repetition penalty (1.05, no
  repeat-ngram 0). Strong penalties push the model off the exact tokens it
  must copy from the user turn.
- The loop guards the final answer three ways (`agent/loop.py`): reject leaked
  control tokens (`_leaked_control_token`), require a numeric tool output to
  appear when one exists (`_observed_numbers`), and on file tasks require a
  touched path or content to appear (`_file_evidence`). A wrong answer must
  never be returned as `passed=True`. Keep these guards.
- Prefer `passed=False` over a false pass. Correctness is judged by the caller;
  the loop's job is to not lie.
- Model soup: `make_soup.py --snapshot-dir checkpoints/snapshots --last 8`.
- Vision SFT: `scripts/train_vision.py` trains the ViT + resampler + projector
  on synthetic coloured shapes (5 colours x 3 shapes). Loss reaches 0.0000 in
  ~3,500 CPU steps. Verify with `scripts/eval_vision.py`; it must beat the
  blank-image baseline by a wide margin. Shapes vary in pixel mask, not just
  position, or the model cannot learn shape.
- Known limit: the 15M model mis-copies operands on unseen arithmetic. This is
  a capacity limit; more data/steps do not fix it. A larger preset would.
- Real small models: `scripts/fetch_model.py` downloads Qwen2.5-0.5B and
  Llama-3.2-1B GGUF weights; `src/smallm/agent/real_brain.py` drives them
  through the same `AgentLoop` via llama.cpp. Run `scripts/aichat.py` to chat,
  `scripts/agent_real.py --eval` to score. A 0.5B-1B model cannot be trained
  from scratch on this box, so we use real public weights and make the *system*
  strong instead.
- Real vision: `src/smallm/agent/vlm.py` wraps SmolVLM-256M-Instruct (a real
  vision-language model) with `VLMBrain.ask(image, question)`. The `see` tool in
  `tools.py` bridges it into the loop, so a text model can read an image in its
  sandbox. Needs transformers, torch, pillow (in `.pylibs`). A 256M VLM takes a
  few seconds per answer on CPU. Score with `scripts/vision_real.py`.
- Small-model loop rules (learned the hard way): never feed the internal
  decision JSON back as an assistant turn, or the model echoes it — describe
  the call in prose. Reject answers that are JSON, carry function-call syntax,
  or leak control tokens. Normalize a messy answer only when it contains a
  number the tools actually produced. Ignore a repeated identical tool call.
