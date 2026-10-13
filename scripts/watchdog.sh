#!/usr/bin/env bash
# Keep a long training run alive across sandbox restarts.
#
#   nohup bash scripts/watchdog.sh .logs/train6.log \
#     python3 scripts/train_real.py --config ... --resume >/dev/null 2>&1 &
#
# Usage: watchdog.sh <logfile> <command...>
set -u
LOG="$1"; shift
CMD=("$@")
cd "$(dirname "$0")/.."
export PYTHONPATH="/workspace/project/.pylibs:src"
mkdir -p "$(dirname "$LOG")"
pid=""
while true; do
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[watchdog $(date -u +%H:%M:%S)] starting: ${CMD[*]}" >> "$LOG"
    "${CMD[@]}" >> "$LOG" 2>&1 &
    pid=$!
  fi
  sleep 30
done
