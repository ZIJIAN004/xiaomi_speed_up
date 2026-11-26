#!/usr/bin/env bash
set -euo pipefail

# Per-slot gradient experiment: train a randomly initialised DSpark draft for the
# ORIGINAL MiniCPM-OCR 1.0 (KIE vocabulary patch) and probe per-slot gradients
# along the way. Outputs go to runs/insight/<run>/ (probe/ holds the measurements).
#
#   bash specdec/insight/run_probe.sh check   # 2 optimizer steps, probes at 0,1,2 (cost + self-checks)
#   bash specdec/insight/run_probe.sh train   # the experiment run (10-epoch LR schedule)
#   bash specdec/insight/run_probe.sh analyze runs/insight/<run>/probe
#
# Extra arguments after the mode go to train_dspark.py. Wrap `train` in tmux.
MODE="${1:?usage: run_probe.sh check|train|analyze [extra args]}"; shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/mr_jiang/models/minicpm-ocr-1.0}"
TARGET="${TARGET:-$MODEL_ROOT/model}"                     # original model, not the LoRA merge
DRAFT_VOCAB="${DRAFT_VOCAB:-$PROJECT_ROOT/runs/dspark/kie_draft_vocab/draft_vocab.pt}"
GPU="${GPU:-1}"
INIT_SEED="${INIT_SEED:-0}"
PROBE_STEPS="${PROBE_STEPS:-0,1,2,5,10,20,50,100,200,400}"
PROBE_EVERY="${PROBE_EVERY:-500}"

cd "$PROJECT_ROOT"
case "$MODE" in
  analyze)
    exec "$PY" specdec/insight/analyze_grad_probe.py "$@" ;;
  check)
    RUN_NAME="${RUN_NAME:-probe_check_$(date +%Y%m%d_%H%M%S)}"
    EXTRA=(--stop-after-steps 2 --probe-steps 0,1,2 --eval-limit 32 --tracker none) ;;
  train)
    RUN_NAME="${RUN_NAME:-probe_random_init_s${INIT_SEED}_$(date +%Y%m%d_%H%M%S)}"
    EXTRA=(--probe-steps "$PROBE_STEPS" --probe-every "$PROBE_EVERY") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
OUT="$PROJECT_ROOT/runs/insight/$RUN_NAME"
# run_dspark.sh supplies the data paths, the GPU 0 guard, the WSL keepalive and the log.
FT_MODEL="$TARGET" DRAFT_VOCAB="$DRAFT_VOCAB" GPU="$GPU" RUN_NAME="$RUN_NAME" OUT="$OUT" \
  bash specdec/dspark/run_dspark.sh train --random-init --init-seed "$INIT_SEED" --epochs 10 \
  "${EXTRA[@]}" "$@"
