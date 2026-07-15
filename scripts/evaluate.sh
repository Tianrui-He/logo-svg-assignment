#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/evaluate.sh PATH_TO_BEST_CHECKPOINT" >&2
  exit 2
fi

python student_kit/eval_self.py \
  --model ./gemma3-270m-it \
  --adapter "$1" \
  --valid valid.jsonl \
  --output results.json \
  --max-new-tokens 2048 \
  --seed 42
