#!/usr/bin/env bash
# Durable, resumable end-to-end pipeline for the brand-new SmallM model.
#
#   bash scripts/pipeline_smallm.sh
#
# Run it under scripts/watchdog.sh so a sandbox restart resumes automatically.
# Every stage is idempotent: a completed stage is skipped, an interrupted stage
# resumes from its last.pt.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="/workspace/project/.pylibs:src"
mkdir -p .logs
L=".logs/pipeline30.log"
say() { echo "[pipeline $(date -u +%H:%M:%S)] $*" >> "$L"; }

PRE=checkpoints/smallm_30m
INS=checkpoints/smallm_30m_instruct
AGT=checkpoints/smallm_30m_agent

say "=== pipeline start ==="

# 1. Pretrain from scratch on real TinyStories tokens.
if [ ! -f "$PRE/model.pt" ]; then
  say "stage 1: pretrain 30M"
  python3 scripts/train_real.py --config configs/smallm_30m.json \
    --data-dir data/processed --out "$PRE" \
    --steps 5000 --batch-size 16 --seq-len 256 --accum 2 --lr 6e-4 \
    --save-every 200 --snapshot-dir checkpoints/smallm_30m_snaps --snapshot-every 250 \
    --resume >> "$L" 2>&1
fi

# 2. Instruction SFT on real human-written Dolly-15k.
if [ ! -f "$INS/model.pt" ]; then
  say "stage 2: instruct SFT (real Dolly)"
  python3 scripts/sft_instruct.py --init "$PRE/model.pt" --out "$INS" \
    --steps 1500 --batch-size 8 --lr 1e-4 --save-every 300 --resume >> "$L" 2>&1
fi

# 3. Agentic SFT: think / tool / verify traces.
if [ ! -f "$AGT/model.pt" ]; then
  say "stage 3: agentic SFT"
  python3 scripts/sft_agentic.py --init "$INS/model.pt" --out "$AGT" \
    --steps 1200 --batch-size 8 --lr 1e-4 --save-every 300 --resume >> "$L" 2>&1
fi

# 4. Model soup over the pretraining snapshots.
if [ ! -f checkpoints/smallm_30m_soup/model.pt ]; then
  say "stage 4: model soup"
  python3 scripts/make_soup.py --snapshot-dir checkpoints/smallm_30m_snaps --last 8 \
    --out checkpoints/smallm_30m_soup >> "$L" 2>&1
fi

say "=== pipeline done ==="
