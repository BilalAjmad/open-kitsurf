# SmallM — a small multimodal, agentic, self-verifying language model

SmallM is a from-scratch implementation of a **small (≈600M–1B parameter)**
vision-language model designed for **fully agentic** operation: it thinks
before it acts, calls tools, drives a sandboxed "computer", checks its own
output, and answers in strict English.

This repository contains the **model, tokenizer, agent, and distillation
training code**. It is a complete, runnable system. It is honest about one
thing: training the weights to frontier quality requires GPUs that are not
present in the development sandbox (4 CPU cores, 15 GB RAM, no GPU). Everything
here is verified by an automated test suite that runs on CPU.

## What is real and tested

| Capability | Status | Where |
|---|---|---|
| Decoder-only transformer (RoPE, GQA, SwiGLU, RMSNorm) | works, tested | `src/smallm/model/` |
| Optional sparse mixture-of-experts | works, tested | `src/smallm/model/mlp.py` |
| Vision encoder (ViT) + perceiver resampler + MLP projector | works, tested | `src/smallm/model/vision.py` |
| Multimodal input splicing at `<image>` tokens | works, tested | `src/smallm/model/text_model.py` |
| KV-cache generation | works, tested | `src/smallm/model/multimodal.py` |
| Byte-level BPE tokenizer with agentic special tokens | works, tested | `src/smallm/tokenizer/bpe.py` |
| Reasoning tokens ` thinking … </think>` and `<verify>` | works, tested | `src/smallm/agent/model_brain.py` |
| Agentic loop: think → act → observe → verify | works, tested | `src/smallm/agent/loop.py` |
| Sandboxed virtual computer (path + command allow-list) | works, tested | `src/smallm/agent/computer.py` |
| Self-verification with self-correction | works, tested | `src/smallm/agent/verifier.py` |
| Strict-English enforcement and correction | works, tested | `src/smallm/agent/strict_english.py` |
| Teacher→student distillation (soft + hard loss) | works, tested | `src/smallm/train/distill.py` |
| Training loop + optimizer | works, tested | `src/smallm/train/pretrain.py` |

Run the tests:

```bash
pip install -r requirements.txt pytest
PYTHONPATH=src python3 -m pytest tests/ -q
```

Run the end-to-end demo:

```bash
PYTHONPATH=src python3 scripts/demo.py
```

## Architecture

```
image ─► ViT encoder ─► perceiver resampler ─► MLP projector ─┐
                                                              ▼
text ─► token embedding ─► [ <image> slots replaced ] ─► transformer decoder ─► logits
                                                              │
                                                              ▼
                                     thinking → tool call → observe → verify → answer
```

- **Text backbone**: pre-norm decoder, grouped-query attention, RoPE, SwiGLU.
- **Vision**: patch embedding, transformer blocks, a fixed number of learned
  latents that resample the image into a small set of visual tokens, then a
  two-layer MLP into text space. Swap the tower for SigLIP/CLIP for production.
- **Two presets**: `configs/smallm_600m.json` (≈606M, dense) and
  `configs/smallm_1b.json` (≈991M, 8-expert MoE, 2 active).

## The agent loop

```
perceive → think → act(tool) → observe → … → answer → verify
                                             ▲            │
                                             └── retry ───┘
```

The loop feeds verification failures back to the model so it can correct itself
before returning. Verification is layered: deterministic English checks plus
optional task checkers (unit tests, exact match, schema). This is how a small
model compensates for its size — strong external checks around a compact model.

## Distillation

`src/smallm/train/distill.py` combines:

- **soft labels**: KL divergence between teacher and student next-token
  distributions at temperature `T`
- **hard labels**: cross-entropy on the teacher's reasoning traces

The teacher only needs to expose `logits_for(input_ids)`. It can be a local
larger model, or a cached logit dump. This is the intended path to make a
600M/1B student punch above its parameter count on narrow, verifiable tasks.

## Scaling to real training (what you need beyond this repo)

1. A GPU cluster. A 1B model needs roughly 20–60B tokens of high-quality,
   license-clean text plus curated multimodal pairs.
2. A teacher model and a distillation corpus with reasoning traces.
3. An evaluation harness. Judge the model on verifiable tasks (code tests,
   tool-use success, exact match), not vibes.

## Honest limitations

- No trained weights ship here. The architecture and recipes are real; the
  weights are not, because that needs a cluster.
- A 1B model will **not** beat frontier 400B+ models in general. It can be
  competitive *per parameter* on narrow, verifiable, distilled tasks.
- The sandbox `screenshot` is textual (headless environment has no display).
- The synthetic data in `src/smallm/data/` exists for tests and smoke runs,
  not for real training.

## Layout

```
configs/           600M and 1B model presets
src/smallm/model/  transformer, vision, multimodal top-level
src/smallm/agent/  loop, tools, computer sandbox, verifier, strict English
src/smallm/tokenizer/  byte-level BPE with agentic tokens
src/smallm/train/  training loop and distillation
src/smallm/data/   synthetic data for tests
tests/             20 CPU tests covering every component
scripts/demo.py    end-to-end walkthrough
```
