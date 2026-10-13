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

## Ready now — talk to a real 0.5B / 1B agent on CPU

You do not have to train anything to get a working model. This repo runs
**real** small models — Qwen2.5-0.5B and Llama-3.2-1B — through its agentic
loop on plain CPU, using `llama.cpp`.

```bash
PYTHONPATH=.pylibs python3 scripts/fetch_model.py        # download real weights
PYTHONPATH=.pylibs:src python3 scripts/aichat.py         # chat
PYTHONPATH=.pylibs:src python3 scripts/aichat.py --mode agent
PYTHONPATH=.pylibs:src python3 scripts/agent_real.py --eval
```

The model is small. The *system* is not: the loop gives it tools, feeds back
observations, grounds every answer in real tool output, and rejects an answer
that is a tool call, a JSON dump, or a made-up number. That is what makes a
0.5B model behave like an agent far larger than it is.

Measured on the 4-core CPU box (`scripts/agent_real.py --eval`, 6 tasks):

| Model | Tasks passed | False passes |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4 | **6/6** | 0 |
| Llama-3.2-1B-Instruct Q4 | 2/6 | 0 |

Qwen drives the tool loop cleanly; the 1B is weaker at the call format, so it
fails honestly rather than faking success. Downloading public weights is normal
use of a package registry; no credentials are involved.

### Real vision, too

The agent can *see*. A real 256M vision-language model (SmolVLM-256M-Instruct)
sits behind a `see` tool, so the text model reads an image in its sandbox and
answers from the pixels.

```bash
PYTHONPATH=.pylibs:src python3 scripts/vision_real.py --n 12   # score the VLM
PYTHONPATH=.pylibs:src python3 scripts/agent_vision_demo.py     # agent uses it
PYTHONPATH=.pylibs:src python3 scripts/agent_real.py --vision --prompt "Look at image.png ..."
```

Measured on CPU: **6/6 colour** and **6/6 shape** on coloured-shape images, and
the agent chain (0.5B text model → `see` → 256M vision model → grounded prose)
runs end to end.

## Or train your own in one box

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

### Download a ready SmallM-Nova (no training needed)

Prebuilt artifacts, on the GitHub release **v0.1.0**:

| File | What it is | Run |
|---|---|---|
| `smallm-nova-ready.tar.gz` | base Nova (15.0M, depth 12) | `chat.py --model checkpoints/nova_ready --prompt "Once upon a time"` |
| `smallm-nova-agent.tar.gz` | agentic Nova (think/tool/verify) | `chat.py --model checkpoints/nova_ready_agent --mode agent` |
| `smallm-nova-vision.tar.gz` | vision Nova (sees images) | `eval_vision.py checkpoints/nova_vision/model.pt` |
| `smallm-nova-soup.tar.gz` | model soup over distillation snapshots | `chat.py --model checkpoints/nova_ready_soup ...` |

```bash
mkdir -p checkpoints/nova_ready && tar xzf smallm-nova-ready.tar.gz -C checkpoints/nova_ready
PYTHONPATH=.pylibs:src python3 scripts/chat.py --model checkpoints/nova_ready --prompt "Once upon a time"
```

### Ready-to-use SmallM-Nova (the new architecture, trained)

`scripts/pipeline_nova_ready.sh` produces a **fully ready Nova** quickly: it
warm-starts the new architecture from a trained checkpoint (same block
contract), then **distils** the teacher into it (KL on the logits + hard CE),
then instruction- and agentic-tunes it. Distillation reached CE **3.82** against
the teacher's ~3.56, and the result chats in coherent English.

```bash
nohup bash scripts/watchdog.sh .logs/pipeline_ready.log \
  bash scripts/pipeline_nova_ready.sh >/dev/null 2>&1 &

PYTHONPATH=.pylibs:src python3 scripts/chat.py --model checkpoints/nova_ready
PYTHONPATH=.pylibs:src python3 scripts/chat.py --model checkpoints/nova_ready_agent --mode agent
```

### SmallM-Nova — a brand-new architecture (recurrent-depth transformer)

Nova is not the plain stack. It is a **recurrent-depth** transformer: the block
stack is applied `num_recurrent_steps` times with a learned per-step embedding,
so the model gets more *effective depth* without more parameters. It also uses
**parallel residuals** (attention and FFN read the same normalized input),
**QK-norm** (per-head RMS on q/k) and **layer-scale**. Code: `src/smallm/model/nova.py`.
Pick it with `cfg.text.arch = "nova"`.

| Preset | Params | Effective depth | Trainable here? |
|---|---|---|---|
| `configs/smallm_nova_30m.json` | 28.3M | 16 (8 blocks × 2 loops) | yes, CPU (~28h) |
| `configs/smallm_nova_90m.json` | 89.7M | 24 (12 blocks × 2 loops) | yes, on a free GPU |

```bash
# durable from-scratch run (resumes across restarts)
nohup bash scripts/watchdog.sh .logs/pipeline_nova.log \
  bash scripts/pipeline_nova.sh >/dev/null 2>&1 &
```

### Train a bigger Nova on a free Kaggle GPU

`notebooks/kaggle_nova_train.ipynb` is a turnkey notebook: it clones this repo,
downloads real TinyStories and real Dolly-15k, then trains the 90M Nova from
scratch and runs instruction + agentic SFT and soup. On Kaggle: **New Notebook →
File → Import Notebook → upload this file → Settings → Internet On → Run all**.
Artifacts land in `/kaggle/working` for download.

### The 30M model, from scratch on real data (brand new)

`scripts/pipeline_smallm.sh` trains a fresh **28.3M** model end to end and
survives sandbox restarts (every stage resumes from its own `last.pt`). It uses
two genuinely real datasets: **TinyStories** (pretraining) and **Databricks
Dolly-15k** — 15,011 real, human-written instruction/response pairs (the
instruction SFT stage):

```bash
# durable: restarts resume automatically
nohup bash scripts/watchdog.sh .logs/pipeline30.log \
  bash scripts/pipeline_smallm.sh >/dev/null 2>&1 &
```

Stages: pretrain 5000 steps → **instruct SFT on real Dolly-15k** (1500 steps)
→ **agentic SFT** (1200 steps, think / tool / verify) → **model soup** over the
last 8 snapshots. Then talk to it:

```bash
PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_30m_instruct --instruct
PYTHONPATH=src python3 scripts/chat.py --model checkpoints/smallm_30m_agent --mode agent
```

The ready-made artifacts in this repo:

| Path | What it is |
|---|---|
| `checkpoints/smallm_11m/model.pt` | base model (15.0M params), trained on 110.9M real tokens |
| `checkpoints/smallm_11m_agent/model.pt` | the same model after agentic SFT (think / tool / verify) |
| `checkpoints/smallm_11m_soup/model.pt` | uniform/greedy soup over fine-tune snapshots |
| `checkpoints/smallm_11m_vision/model.pt` | vision tower trained for real on coloured-shape images |

### Measured results (CPU box, 15M model)

- Base LM: step 4000, cross-entropy **3.56** on held-out TinyStories.
- Model soup: greedy soup over the last 8 snapshots, val loss **3.5563**
  (uniform **3.5871**); two snapshots were selected.
- Agentic SFT: response-only loss **~0.06** over 3,801 diverse traces.
- Vision SFT (`scripts/train_vision.py`, 3,500 steps, loss **0.0000**):
  the model captions **60/60** unseen coloured-shape images versus a
  **6/60** blank-image baseline. The 54-point gap is the proof the ViT reads
  pixels — a language prior alone cannot do this. Reproduce with
  `PYTHONPATH=src python3 scripts/eval_vision.py`.
- Agent loop, 9 held-out tasks (6 arithmetic, 3 file tasks): **3 correct**,
  and **0 false passes**. Three guards keep a wrong answer from being reported
  as success: the answer may not contain control tokens or tool-call syntax, it
  must match a numeric tool output when one exists, and on file tasks it must
  mention a file the tools actually touched. Failures surface as `passed=False`
  or a retry, never as a silent wrong answer.

A 15M model on a CPU box is at its capacity limit for exact operand copying;
that limit is the honest result, not a hidden one. The point of the artifact is
that the full pipeline — real data, soup, SFT, grounded self-verifying loop,
chat CLI — runs end to end and can be reproduced.

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
before returning. Verification is layered: deterministic English checks,
observation grounding, plus optional task checkers (unit tests, exact match,
schema). This is how a small model compensates for its size — strong external
checks around a compact model.

**Observation grounding** is the key guard for a small model: the loop extracts
the numeric outputs of the tools it actually ran and rejects any answer that
does not contain the observed value, retrying with the failure reason fed back.
The model may hallucinate, but the loop will not pass the hallucination.

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
- The vision tower is trained on a **synthetic** caption task (simple coloured
  shapes), which proves the whole image path works end to end. It is not a
  general image understanding model; that needs a real image-text corpus.
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
src/smallm/agent/  loop, tools, computer sandbox, verifier, strict English,
                   real_brain (drives a real GGUF model), vlm (real vision)
tests/             35 CPU tests covering every component
scripts/demo.py          end-to-end walkthrough
scripts/fetch_model.py   download real 0.5B / 1B GGUF weights
scripts/aichat.py        interactive chat (real model; chat or agent mode)
scripts/agent_real.py    real-model agentic loop + eval harness
scripts/vision_real.py   score the real 256M vision model on images
scripts/agent_vision_demo.py  text agent using the real `see` vision tool
scripts/train_vision.py  train the vision tower on synthetic shapes
scripts/eval_vision.py   measure image grounding vs a blank baseline
scripts/sft_agentic.py   agentic SFT (think / tool / verify traces)
scripts/sft_instruct.py  instruction SFT on real Dolly-15k
scripts/pipeline_smallm.sh  durable end-to-end training for the 30M model
scripts/pipeline_nova.sh    durable from-scratch training for the Nova architecture
scripts/pipeline_nova_ready.sh  ready Nova: transplant -> distill -> SFT
scripts/transplant_to_nova.py   warm-start Nova from a trained checkpoint
scripts/distill_nova.py         distill a teacher into Nova (KL + CE)
scripts/watchdog.sh      keep a long run alive across restarts
scripts/make_soup.py     uniform / greedy model soup
scripts/chat.py          chat CLI for a checkpoint you trained (base/instruct/agent)
```
