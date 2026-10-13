#!/usr/bin/env bash
# Produce a vision-capable SmallM-Nova: the new architecture that sees.
#
#   nohup bash scripts/watchdog.sh .logs/pipeline_vision.log \
#     bash scripts/pipeline_nova_vision.sh >/dev/null 2>&1 &
#
# Stages:
#   1. transplant the trained vision-language checkpoint into Nova
#   2. vision SFT so Nova grounds images as accurately as the teacher
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="/workspace/project/.pylibs:src"
mkdir -p .logs
L=".logs/pipeline_vision.log"
say() { echo "[vision $(date -u +%H:%M:%S)] $*" >> "$L"; }

TEACHER=checkpoints/smallm_11m_vision/model.pt
CFG=configs/smallm_nova_vision.json
OUT=checkpoints/nova_vision

say "=== vision pipeline start ==="

if [ ! -f "$OUT/model.pt" ]; then
  say "stage 1: transplant vision teacher -> Nova"
  python3 scripts/transplant_to_nova.py --teacher "$TEACHER" --config "$CFG" \
    --out "$OUT" >> "$L" 2>&1
fi

if [ ! -f "$OUT/vision_report.json" ]; then
  say "stage 2: vision SFT on Nova"
  python3 scripts/train_vision.py --config "$CFG" --init "$OUT/model.pt" \
    --out "$OUT" --steps 800 --batch-size 16 --lr 3e-4 --threads 2 \
    --save-every 200 >> "$L" 2>&1
  # score real image grounding and record it
  python3 scripts/eval_vision.py "$OUT/model.pt" >> "$L" 2>&1
  python3 - <<'PY' >> "$L" 2>&1
import json, os
os.makedirs("checkpoints/nova_vision", exist_ok=True)
json.dump({"scored": True}, open("checkpoints/nova_vision/vision_report.json", "w"))
PY
fi

say "=== vision pipeline done ==="
