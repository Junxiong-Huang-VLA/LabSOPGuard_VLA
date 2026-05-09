#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/LabSOPGuard}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_YAML="${DATASET_YAML:-data/dataset/dataset.yaml}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL="${MODEL:-yolo26s.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-yolo26s_autodl_8_1_1}"

cd "$PROJECT_ROOT"

mkdir -p .ultralytics .matplotlib outputs/training
export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
export MPLCONFIGDIR="$PROJECT_ROOT/.matplotlib"

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -r requirements.txt

DATASET_ROOT_ARGS=()
if [[ -n "$DATASET_ROOT" ]]; then
  DATASET_ROOT_ARGS=(--dataset-root "$DATASET_ROOT")
fi

"$PYTHON_BIN" scripts/train_yolo_lab.py \
  --dataset-yaml "$DATASET_YAML" \
  "${DATASET_ROOT_ARGS[@]}" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --project outputs/training \
  --name "$RUN_NAME"

echo "[DONE] training output: $PROJECT_ROOT/outputs/training/$RUN_NAME"
