#!/usr/bin/env bash
# Train the GR00T baseline on a deformablebench LeRobot v3.0 dataset.
#
# Thin facade over the submodule's finetune.sh — it only fills in the
# deformablebench-specific defaults (dual-piper embodiment + modality config) and
# forwards everything else. All GR00T training logic lives in the submodule.
#
# RUN THIS IN THE GR00T ENV (py3.10 uv venv), NOT deformablebench's .venv. Full guide
# (env setup, data prep, tuning knobs): see README.md next to this file.
#
# Usage:
#   BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B \
#   DATASET_PATH=/path/to/data/lerobot_dataset/<task> \
#   OUTPUT_DIR=/path/to/out \
#   [NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=32 USE_WANDB=1] \
#   bash train.sh [-- <extra finetune.sh args>]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
GR00T="${GR00T_DIR:-$ROOT/thirdparty/isaac_gr00t}"

: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH to the GR00T-N1.7-3B checkpoint dir}"
: "${DATASET_PATH:?set DATASET_PATH to a deformablebench LeRobot v3.0 dataset dir}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for checkpoints/logs}"

MODALITY_SRC="$GR00T/examples/DeformableBimanual/modality.json"
MODALITY_CFG="${MODALITY_CONFIG_PATH:-$GR00T/examples/DeformableBimanual/dual_piper_config.py}"

# GR00T needs the dual-piper modality.json inside the dataset's meta/. We do NOT
# mutate your dataset silently — if it's missing, tell you exactly how to add it
# (dropping this additive metadata file in, or building a symlinked GR00T view).
if [ ! -f "$DATASET_PATH/meta/modality.json" ]; then
  echo "ERROR: $DATASET_PATH/meta/modality.json is missing." >&2
  echo "GR00T reads modality.json to map dataset columns -> model modalities." >&2
  echo "Add it (additive, does not touch your robot data):" >&2
  echo "    cp $MODALITY_SRC $DATASET_PATH/meta/modality.json" >&2
  exit 1
fi

exec bash "$GR00T/examples/finetune.sh" \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --embodiment-tag new_embodiment \
  --modality-config-path "$MODALITY_CFG" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
