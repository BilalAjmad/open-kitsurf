#!/usr/bin/env bash
# Build the full Ageis family from scratch: base -> agent SFT -> vision SFT.
#
#   bash scripts/pipeline_ageis.sh base      # train the base model (long)
#   bash scripts/pipeline_ageis.sh agent     # agentic SFT on top of base
#   bash scripts/pipeline_ageis.sh vision    # vision SFT on top of base
#   bash scripts/pipeline_ageis.sh all       # everything, in order
#
# Each stage is resumable: re-run it and it continues from last.pt.
set -u
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/.pylibs:$PWD/src"
PY=python3
STAGE="${1:-all}"

base() {
  echo "=== [1/3] train Ageis base (from scratch) ==="
  $PY scripts/train_real.py \
    --config configs/smallm_ageis_15m.json \
    --data-dir data/processed --out checkpoints/ageis_15m \
    --steps 3000 --batch-size 8 --seq-len 256 --accum 2 \
    --lr 6e-4 --warmup 200 --log-every 25 --save-every 100 \
    --snapshot-dir checkpoints/ageis_15m_snaps --snapshot-every 200 \
    --threads 4 --resume
}

agent() {
  echo "=== [2/3] agentic SFT (think / tool / verify) ==="
  $PY scripts/sft_agentic.py \
    --init checkpoints/ageis_15m/model.pt \
    --tokenizer data/processed/tokenizer.json \
    --out checkpoints/ageis_agent --steps 600 --batch-size 8 \
    --seq-len 256 --lr 1e-4 --threads 4 --save-every 200
}

vision() {
  echo "=== [3/3] vision SFT (image grounding) ==="
  $PY scripts/train_vision.py \
    --init checkpoints/ageis_15m/model.pt \
    --config configs/smallm_ageis_vision.json \
    --tokenizer data/processed/tokenizer.json \
    --out checkpoints/ageis_vision --steps 2000 --batch-size 16 \
    --lr 3e-4 --threads 4 --save-every 500
}

case "$STAGE" in
  base) base ;;
  agent) agent ;;
  vision) vision ;;
  all) base && agent && vision ;;
  *) echo "usage: $0 {base|agent|vision|all}"; exit 1 ;;
esac
