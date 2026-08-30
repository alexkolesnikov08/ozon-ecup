#!/usr/bin/env bash
set -euo pipefail

# Environment overrides:
#   SEGMENTED_PYTHON=.venv/bin/python
#   SEGMENTED_THREADS=16
#   SEGMENTED_BATCH_SIZE=50000
#   SEGMENTED_QUICK=1
#   SEGMENTED_OVERWRITE=1

SEGMENTED_PYTHON="${SEGMENTED_PYTHON:-.venv/bin/python}"
SEGMENTED_THREADS="${SEGMENTED_THREADS:-16}"
SEGMENTED_BATCH_SIZE="${SEGMENTED_BATCH_SIZE:-50000}"
SEGMENTED_QUICK="${SEGMENTED_QUICK:-0}"
SEGMENTED_OVERWRITE="${SEGMENTED_OVERWRITE:-0}"
SEGMENTED_STAGE="${1:-all}"

if [[ ! -x "$SEGMENTED_PYTHON" ]]; then
  echo "Python executable not found: $SEGMENTED_PYTHON" >&2
  echo "Create .venv and install requirements_segmented.txt first." >&2
  exit 2
fi

build_args=(
  src/build_segmented_features.py
  --data data/train.parquet
  --sample sample_submit.csv
  --out-dir data/segmented_base
  --batch-size "$SEGMENTED_BATCH_SIZE"
)
if [[ "$SEGMENTED_OVERWRITE" == "1" ]]; then
  build_args+=(--overwrite)
fi

train_args=(
  src/train_segmented_submit.py
  --features-dir data/segmented_base
  --sample sample_submit.csv
  --submission submissions/submission_segmented.csv
  --report reports/segmented_pipeline.json
  --threads "$SEGMENTED_THREADS"
)
if [[ "$SEGMENTED_QUICK" == "1" ]]; then
  train_args+=(--quick)
fi

case "$SEGMENTED_STAGE" in
  all)
    "$SEGMENTED_PYTHON" "${build_args[@]}"
    "$SEGMENTED_PYTHON" "${train_args[@]}"
    ;;
  features)
    "$SEGMENTED_PYTHON" "${build_args[@]}"
    ;;
  train)
    "$SEGMENTED_PYTHON" "${train_args[@]}"
    ;;
  *)
    echo "Usage: bash scripts/run_segmented_pipeline.sh [all|features|train]" >&2
    exit 2
    ;;
esac
