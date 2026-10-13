#!/usr/bin/env bash
# Durable, resumable pipeline for the brand-new Nova architecture.
#
#   nohup bash scripts/watchdog.sh .logs/pipeline_nova.log \
#     bash scripts/pipeline_nova.sh >/dev/null 2>&1 &
#
# Nova = recurrent-depth transformer (the stack is applied num_recurrent_steps
# times with a learned per-step embedding), parallel residuals, QK-norm and
# layer-scale. 28.3M params, effective depth 16 (8 unique blocks x 2 loops).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="/workspace/project/.pylibs:src"
mkdir -p .logs
L=".logs/pipeline_nova.log"
say() { echo "[nova $(date -u +%H:%M:%S)] $*" >> "$L"; }

CFG=configs/smallm_nova_30m.json
PRE=checkpoints/nova_30m
INS=checkpoints/nova_30m_instruct
AGT=checkpoints/nova_30m_agent
SNAP=checkpoints/nova_30m_snaps

say "=== nova pipeline start ==="

# 1. Pretrain from scratch on real TinyStories tokens.
if [ ! -f "$PRE/model.pt" ]; then
  say "stage 1: pretrain nova 28.3M"
  python3 scripts/train_real.py --config "$CFG" \
    --data-dir data/processed --out "$PRE" \
    --steps 5000 --batch-size 16 --seq-len 256 --accum 2 --lr 6e-4 \
    --save-every 200 --snapshot-dir "$SNAP" --snapshot-every 250 \
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
if [ ! -f checkpoints/nova_30m_soup/model.pt ]; then
  say "stage 4: model soup"
  python3 scripts/make_soup.py --snapshot-dir "$SNAP" --last 8 \
    --out checkpoints/nova_30m_soup >> "$L" 2>&1
fi

say "=== nova pipeline done ==="
