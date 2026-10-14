#!/usr/bin/env bash
# End-to-end build of the SmallM-Chimera family, from raw text to usable models.
#
#   bash scripts/pipeline_chimera.sh curate    # curate + tokenize the corpus
#   bash scripts/pipeline_chimera.sh pretrain  # train the base model
#   bash scripts/pipeline_chimera.sh agent     # agentic SFT
#   bash scripts/pipeline_chimera.sh vision    # vision SFT
#   bash scripts/pipeline_chimera.sh soup      # greedy model soup
#   bash scripts/pipeline_chimera.sh all       # everything, in order
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/.pylibs:$PWD/src"
PY=python3

curate() {
  $PY scripts/curate_data.py --raw data/raw/tinystories.txt \
      --out data/curated/stories.jsonl --report data/curated/stories_stats.json
  $PY scripts/build_instruct_corpus.py \
      --raw data/raw/dolly/databricks-dolly-15k.jsonl \
      --out data/curated/instruct.jsonl
  cat data/curated/stories.jsonl data/curated/instruct.jsonl > data/curated/all.txt
  mkdir -p data/processed_chimera
  cp data/processed/tokenizer.json data/processed_chimera/tokenizer.json
  $PY scripts/prepare_data.py --raw data/curated/all.txt \
      --out-dir data/processed_chimera --allow-special
}

pretrain() {
  $PY scripts/train_real.py --config configs/smallm_chimera_15m.json \
      --data-dir data/processed_chimera --out checkpoints/chimera_15m \
      --steps 4000 --batch-size 8 --seq-len 256 --accum 2 --lr 6e-4 \
      --warmup 200 --log-every 25 --save-every 200 \
      --snapshot-dir checkpoints/chimera_15m_snaps --snapshot-every 500 --threads 4
}

agent() {
  $PY scripts/sft_agentic.py --init checkpoints/chimera_15m/model.pt \
      --tokenizer data/processed_chimera/tokenizer.json \
      --out checkpoints/chimera_agent --steps 600 --batch-size 8 \
      --seq-len 256 --lr 1e-4 --threads 4
}

vision() {
  $PY scripts/train_vision.py --init checkpoints/chimera_15m/model.pt \
      --config configs/smallm_chimera_vision.json \
      --tokenizer data/processed_chimera/tokenizer.json \
      --out checkpoints/chimera_vision --steps 2000 --batch-size 16 --lr 3e-4 --threads 4
}

soup() {
  $PY scripts/make_soup.py --snapshot-dir checkpoints/chimera_15m_snaps \
      --last 8 --data-dir data/processed_chimera --out checkpoints/chimera_15m_soup --threads 4
}

case "${1:-all}" in
  curate) curate ;;
  pretrain) pretrain ;;
  agent) agent ;;
  vision) vision ;;
  soup) soup ;;
  all) curate; pretrain; agent; vision; soup ;;
  *) echo "usage: $0 {curate|pretrain|agent|vision|soup|all}"; exit 2 ;;
esac
echo "chimera pipeline: $1 done"
