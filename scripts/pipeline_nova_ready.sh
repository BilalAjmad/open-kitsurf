#!/usr/bin/env bash
# Produce a fully ready SmallM-Nova: the new architecture, warm-started from a
# trained teacher, distilled, then instruction- and agentic-tuned.
#
#   nohup bash scripts/watchdog.sh .logs/pipeline_ready.log \
#     bash scripts/pipeline_nova_ready.sh >/dev/null 2>&1 &
#
# Stages (each resumable / idempotent):
#   1. transplant trained weights into Nova (instant warm start)
#   2. distill teacher -> Nova (KL + CE) so Nova reproduces the teacher
#   3. instruction SFT on real Dolly-15k
#   4. agentic SFT (think / tool / verify)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="/workspace/project/.pylibs:src"
mkdir -p .logs
L=".logs/pipeline_ready.log"
say() { echo "[ready $(date -u +%H:%M:%S)] $*" >> "$L"; }

TEACHER=checkpoints/smallm_11m/model.pt
CFG=configs/smallm_nova_ready.json
NOVA=checkpoints/nova_ready
INS=checkpoints/nova_ready_instruct
AGT=checkpoints/nova_ready_agent

say "=== ready pipeline start ==="

if [ ! -f "$NOVA/model.pt" ]; then
  say "stage 1: transplant teacher -> Nova"
  python3 scripts/transplant_to_nova.py --teacher "$TEACHER" --config "$CFG" \
    --out "$NOVA" >> "$L" 2>&1
fi

if [ ! -f "$NOVA/distill_report.json" ]; then
  say "stage 2: distillation"
  python3 scripts/distill_nova.py --teacher "$TEACHER" --init "$NOVA/model.pt" \
    --data-dir data/processed --out "$NOVA" --steps 600 --batch-size 8 --seq-len 192 \
    --lr 3e-4 --warmup 40 --alpha 0.5 --save-every 100 --threads 4 --resume >> "$L" 2>&1
fi

if [ ! -f "$INS/model.pt" ]; then
  say "stage 3: instruct SFT (real Dolly)"
  python3 scripts/sft_instruct.py --init "$NOVA/model.pt" --out "$INS" \
    --steps 900 --batch-size 8 --lr 8e-5 --save-every 300 --resume >> "$L" 2>&1
fi

if [ ! -f "$AGT/model.pt" ]; then
  say "stage 4: agentic SFT"
  python3 scripts/sft_agentic.py --init "$INS/model.pt" --out "$AGT" \
    --steps 900 --batch-size 8 --lr 8e-5 --save-every 300 --resume >> "$L" 2>&1
fi

say "=== ready pipeline done ==="
