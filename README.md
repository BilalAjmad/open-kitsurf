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

## Ready to use — train your own in one box

The repository ships a complete, tested pipeline that turns real public data
into a working chat + agent model on CPU:

```bash
# 1. real data (TinyStories), train a real BPE, tokenize a 100M+ token corpus
curl -sL -r 0-314572799 -o data/raw/tinystories.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
PYTHONPATH=src python3 scripts/prepare_data.py \
  --raw data/raw/tinystories.txt --out-dir data/processed \
  --vocab-size 4096 --tokenizer-sample-mb 20 --tokenize-limit-mb 250

# 2. pretrain (checkpoints every N steps, resumable)
PYTHONPATH=src python3 scripts/train_real.py \
  --config configs/smallm_11m.json --data-dir data/processed \
  --out checkpoints/smallm_11m --steps 1200 --batch-size 32 --seq-len 256

# 3. teach it the agentic format (thinking, tools, self-verification)
PYTHONPATH=src python3 scripts/sft_agentic.py \
  --init checkpoints/smallm_11m/model.pt --out checkpoints/smallm_11m_agent \
  --steps 400 --batch-size 8 --seq-len 256

# 4. model soup: merge several fine-tunes (uniform + greedy)
#    (train with --snapshot-dir to emit soup candidates)
PYTHONPATH=src python3 scripts/train_real.py \
  --config configs/smallm_11m.json --data-dir data/processed \
  --out checkpoints/smallm_11m --steps 1500 --lr 1.2e-4 --resume \
  --snapshot-dir checkpoints/snapshots --snapshot-every 75
PYTHONPATH=src python3 scripts/make_soup.py \
  --variants checkpoints/snapshots/*.pt \
  --data-dir data/processed --out checkpoints/smallm_11m_soup

# 5. use it
PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_11m_agent
PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_11m_agent --mode agent
```

This produces a **real trained checkpoint** you can talk to. It is a small
model on a small corpus, so expect the quality of a small model — coherent
short English and working tool traces, not frontier prose.

The ready-made artifacts in this repo:

| Path | What it is |
|---|---|
| `checkpoints/smallm_11m/model.pt` | base model (15.0M params), trained on 110.9M real tokens |
| `checkpoints/smallm_11m_agent/model.pt` | the same model after agentic SFT (think / tool / verify) |
| `checkpoints/smallm_11m_soup/model.pt` | uniform/greedy soup over fine-tune snapshots |

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

- The shipped checkpoint (`checkpoints/smallm_11m`) is a **15M-parameter**
  model trained on one CPU box, not the 600M/1B presets. It is a real,
  ready-to-use artifact that demonstrates the whole pipeline end to end. The
  600M/1B presets are the architecture targets for a GPU cluster.
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
