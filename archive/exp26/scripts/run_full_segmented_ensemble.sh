#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: base features -> classical hierarchy -> weekly Transformer -> blend.
# Override resource settings through the SEGMENTED_* variables below.

SEGMENTED_PYTHON="${SEGMENTED_PYTHON:-.venv/bin/python}"
SEGMENTED_THREADS="${SEGMENTED_THREADS:-16}"
SEGMENTED_BATCH_SIZE="${SEGMENTED_BATCH_SIZE:-50000}"
SEGMENTED_DEVICE="${SEGMENTED_DEVICE:-auto}"
SEGMENTED_EPOCHS="${SEGMENTED_EPOCHS:-12}"
SEGMENTED_TORCH_WORKERS="${SEGMENTED_TORCH_WORKERS:-0}"
SEGMENTED_QUICK="${SEGMENTED_QUICK:-0}"

if [[ ! -x "$SEGMENTED_PYTHON" ]]; then
  echo "Python executable not found: $SEGMENTED_PYTHON" >&2
  exit 2
fi

build_args=(
  archive/exp26/src/build_segmented_features.py
  --data data/train.parquet
  --sample sample_submit.csv
  --out-dir data/segmented_base
  --batch-size "$SEGMENTED_BATCH_SIZE"
)
classical_args=(
  archive/exp26/src/train_segmented_submit.py
  --features-dir data/segmented_base
  --sample sample_submit.csv
  --predictions-dir data/segmented_predictions/classical
  --submission archive/exp26/submissions/submission_segmented.csv
  --report archive/exp26/reports/segmented_pipeline.json
  --threads "$SEGMENTED_THREADS"
)
transformer_args=(
  archive/exp26/src/train_weekly_transformer.py
  --features-dir data/segmented_base
  --predictions-dir data/segmented_predictions/transformer
  --report archive/exp26/reports/weekly_transformer.json
  --device "$SEGMENTED_DEVICE"
  --epochs "$SEGMENTED_EPOCHS"
  --workers "$SEGMENTED_TORCH_WORKERS"
)
if [[ "$SEGMENTED_QUICK" == "1" ]]; then
  classical_args+=(--quick)
  transformer_args+=(--quick)
fi

"$SEGMENTED_PYTHON" "${build_args[@]}"
"$SEGMENTED_PYTHON" "${classical_args[@]}"
"$SEGMENTED_PYTHON" "${transformer_args[@]}"
"$SEGMENTED_PYTHON" archive/exp26/src/blend_segmented_transformer.py \
  --classical-dir data/segmented_predictions/classical \
  --transformer-dir data/segmented_predictions/transformer \
  --features-dir data/segmented_base \
  --sample sample_submit.csv \
  --submission archive/exp26/submissions/submission_segmented_transformer.csv \
  --report archive/exp26/reports/segmented_transformer_blend.json
