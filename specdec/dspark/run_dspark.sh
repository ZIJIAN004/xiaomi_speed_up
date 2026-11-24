#!/usr/bin/env bash
set -euo pipefail

# MiniCPM-OCR 1.0 DSpark pipeline launcher (inside WSL, minicpm_ocr env).
#
#   bash specdec/dspark/run_dspark.sh check   # alignment gate: ORIGINAL model + official draft
#   bash specdec/dspark/run_dspark.sh eval    # official draft on the fine-tuned target (no training)
#   bash specdec/dspark/run_dspark.sh train   # warm-start from the official draft on the fine-tuned target
#   bash specdec/dspark/run_dspark.sh smoke   # train for 2 optimizer steps on 64 samples
#
# Extra arguments after the mode go to the Python script, e.g. `train --epochs 3`.
# Runs in the foreground with a WSL keepalive; wrap it in tmux for long runs.
MODE="${1:?usage: run_dspark.sh check|eval|train|smoke [extra args]}"; shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/mr_jiang/models/minicpm-ocr-1.0}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/model}"
OFFICIAL_DRAFT="${OFFICIAL_DRAFT:-$MODEL_ROOT/dspark}"
FT_MODEL="${FT_MODEL:-/home/mr_jiang/models/minicpm-ocr-1.0-ft/model}"
DATA_DIR="${DATA_DIR:-/home/mr_jiang/Lora/runs/idspace_e3/prepared}"   # JSONL image paths start with images/
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DATA_DIR/validation.jsonl}"
GPU="${GPU:-1}"
RUN_NAME="${RUN_NAME:-dspark_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$PROJECT_ROOT/runs/dspark/$RUN_NAME}"

# Host rule: no job above 15 GB on GPU 0. Draft training is not measured yet.
if [[ "$GPU" == "0" && "${ALLOW_GPU0:-0}" != "1" ]]; then
  echo "refusing GPU 0 (host rule: <=15GB jobs only); set GPU=1 or ALLOW_GPU0=1 after checking memory" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
source "$PROJECT_ROOT/envs/wsl_keepalive.sh"
start_wsl_keepalive "$$" || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpu_before.csv"

DATA_ARGS=(--train-jsonl "$TRAIN_JSONL" --image-root "$DATA_DIR" --val-jsonl "$VAL_JSONL")
cd "$PROJECT_ROOT"
case "$MODE" in
  check)
    "$PY" specdec/dspark/check_alignment.py --model "$BASE_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" "$@" 2>&1 | tee "$OUT/check.log" ;;
  eval)
    "$PY" specdec/dspark/train_dspark.py --eval-only --model "$FT_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" "$@" 2>&1 | tee "$OUT/eval.log" ;;
  train)
    "$PY" specdec/dspark/train_dspark.py --model "$FT_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" "$@" 2>&1 | tee "$OUT/train.log" ;;
  smoke)
    "$PY" specdec/dspark/train_dspark.py --model "${SMOKE_MODEL:-$BASE_MODEL}" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" --max-samples 64 --grad-accum 4 --max-steps 2 --epochs 1 \
      --eval-limit 8 --log-every 1 "$@" 2>&1 | tee "$OUT/smoke.log" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpu_after.csv"
echo "outputs: $OUT"
