# SmallM model card

## What this is

SmallM is a small, fully open, CPU-trainable language model with four
deliberate design choices:

1. **Self-verification.** The model is trained to emit a ` thinking` block and a
   `<verify>` block, so it checks its own answer before returning it.
2. **A computer.** It drives a sandboxed tool registry (files, directories,
   `calc`, `python`) through a real perceive-think-act-observe loop.
3. **Vision.** A ViT-style encoder with a perceiver resampler projects image
   features into the token stream at `<image>` tokens.
4. **Strict English.** A post-processor rewrites contractions and removes
   banned punctuation so the output stays plain and short.

## Honest expectations

This is a small model trained for a short time on one small corpus. It is
**not** competitive with frontier models. It produces coherent short English
and valid agentic traces. It does not reason like a large model, and it will
make mistakes. Treat it as a complete, readable, end-to-end reference you can
run, study, and extend.

Training budget used for the shipped checkpoint: a single CPU box (4 cores,
no GPU), TinyStories (real public data), a 4096-token BPE learned on that data.

## Data

* Source: TinyStories (`roneneldan/TinyStories`, public, synthetic short
  stories written in simple English).
* Tokenized corpus: ~110.9M tokens, stored as a `uint16` memmap.
* Tokenizer: byte-level BPE, vocab 4096, learned on a 20 MB sample.

## Training

* Objective: next-token cross-entropy.
* Optimizer: AdamW with cosine LR, warmup, gradient clipping, accumulation.
* Checkpointing: periodic `last.pt` plus a weights-only `model.pt` export.
* Model soup: uniform and greedy soups over fine-tune variants
  (`scripts/make_soup.py`), keeping the lowest validation loss.

## Files in a checkpoint directory

| File | Purpose |
|---|---|
| `model.pt` | weights + config, ready to load |
| `tokenizer.json` | the BPE used to train it |
| `last.pt` | full training state for resume |
| `train_report.json` | parameter count and final loss |

## Limitations

* Short context (512 tokens by default).
* English only, and only the simple style it was trained on.
* Tool use is limited to the sandboxed registry.
* No safety alignment; do not deploy it against untrusted input.
