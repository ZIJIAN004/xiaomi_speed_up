#!/usr/bin/env bash
set -euo pipefail

# MiniCPM-OCR 1.0 DSpark pipeline launcher (inside WSL, minicpm_ocr env).
#
#   bash specdec/dspark/run_dspark.sh merge   # LoRA adapter + base -> $FT_MODEL, then verify vs PEFT ($MERGE_PY env)
#   bash specdec/dspark/run_dspark.sh vocab   # KIE draft-vocabulary patch -> $DRAFT_VOCAB (CPU)
#   bash specdec/dspark/run_dspark.sh check   # alignment gate: ORIGINAL model + official draft
#   bash specdec/dspark/run_dspark.sh eval    # official draft (unpatched) on the fine-tuned target
#   bash specdec/dspark/run_dspark.sh train   # warm start on the fine-tuned target (all train rows by default)
#   bash specdec/dspark/run_dspark.sh smoke   # train for 2 optimizer steps on 64 samples (with --profile-stages)
#   bash specdec/dspark/run_dspark.sh lengths # token-length stats (processor only, CPU)
#
# Data: the KIE stage of the MiniCPM-OCR LoRA run (layout-region crop, 16-field prompt,
# max_slice_nums=1); train.jsonl trains, validation.jsonl is held out.
# Extra arguments after the mode go to the Python script, e.g. `train --epochs 5`.
# Runs in the foreground with a WSL keepalive; wrap it in tmux for long runs.
MODE="${1:?usage: run_dspark.sh merge|vocab|check|eval|train|smoke|lengths [extra args]}"; shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/mr_jiang/models/minicpm-ocr-1.0}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/model}"
OFFICIAL_DRAFT="${OFFICIAL_DRAFT:-$MODEL_ROOT/dspark}"
LORA_DIR="${LORA_DIR:-/home/mr_jiang/speed_up/model/MiniCPM-OCR-1.0-LoRA-20261123_230559_273596-step6754}"
ADAPTER="${ADAPTER:-$LORA_DIR/adapter}"
FT_MODEL="${FT_MODEL:-$LORA_DIR/merged}"                                   # written by `merge`
MERGE_PY="${MERGE_PY:-/home/mr_jiang/Lora/envs/xiaomi_ocr0/bin/python}"     # env that trained the LoRA (has peft)
DATA_DIR="${DATA_DIR:-/home/mr_jiang/Lora/runs/minicpm_official_9to1_20261123_230559_273596}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DATA_DIR/validation.jsonl}"
LAYOUT_JSONL="${LAYOUT_JSONL:-$DATA_DIR/layout.jsonl}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"                                         # >0: seeded random subset; 0 = all rows
DRAFT_VOCAB="${DRAFT_VOCAB-$PROJECT_ROOT/runs/dspark/kie_draft_vocab/draft_vocab.pt}"   # DRAFT_VOCAB= : official vocab
GPU="${GPU:-1}"
RUN_NAME="${RUN_NAME:-dspark_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$PROJECT_ROOT/runs/dspark/$RUN_NAME}"

# Host rule: no job above 15 GB on GPU 0. Draft training is not measured yet.
if [[ "$GPU" == "0" && "${ALLOW_GPU0:-0}" != "1" ]]; then
  echo "refusing GPU 0 (host rule: <=15GB jobs only); set GPU=1 or ALLOW_GPU0=1 after checking memory" >&2
  exit 2
fi
VOCAB_ARGS=()
if [[ -n "$DRAFT_VOCAB" && ( "$MODE" == "train" || "$MODE" == "smoke" ) ]]; then
  if [[ ! -f "$DRAFT_VOCAB" ]]; then
    echo "missing $DRAFT_VOCAB: run 'run_dspark.sh vocab' first, or set DRAFT_VOCAB= for the official vocab" >&2
    exit 2
  fi
  VOCAB_ARGS=(--draft-vocab "$DRAFT_VOCAB")
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
source "$PROJECT_ROOT/envs/wsl_keepalive.sh"
start_wsl_keepalive "$$" || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpu_before.csv"

DATA_ARGS=(--train-jsonl "$TRAIN_JSONL" --image-root "$DATA_DIR" --val-jsonl "$VAL_JSONL"
           --task kie --layout-jsonl "$LAYOUT_JSONL")
cd "$PROJECT_ROOT"
case "$MODE" in
  merge)
    "$MERGE_PY" specdec/dspark/merge_lora.py --base "$BASE_MODEL" --adapter "$ADAPTER" --output "$FT_MODEL" \
      --data-dir "$DATA_DIR" --layout-jsonl "$LAYOUT_JSONL" --verify-samples 32 "$@" 2>&1 | tee "$OUT/merge.log" ;;
  vocab)
    mkdir -p "$(dirname "$DRAFT_VOCAB")"
    "$PY" specdec/dspark/build_kie_draft_vocab.py --model "$BASE_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      --train-jsonl "$TRAIN_JSONL" --val-jsonl "$VAL_JSONL" --output "${DRAFT_VOCAB%.pt}" "$@" 2>&1 | tee "$OUT/vocab.log" ;;
  check)
    "$PY" specdec/dspark/check_alignment.py --model "$BASE_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" "$@" 2>&1 | tee "$OUT/check.log" ;;
  eval)
    "$PY" specdec/dspark/train_dspark.py --eval-only --model "$FT_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" --output-dir "$OUT" "$@" 2>&1 | tee "$OUT/eval.log" ;;
  train)
    "$PY" specdec/dspark/train_dspark.py --model "$FT_MODEL" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" "${VOCAB_ARGS[@]}" --train-samples "$TRAIN_SAMPLES" --output-dir "$OUT" "$@" \
      2>&1 | tee "$OUT/train.log" ;;
  smoke)
    "$PY" specdec/dspark/train_dspark.py --model "${SMOKE_MODEL:-$BASE_MODEL}" --init-draft "$OFFICIAL_DRAFT" \
      "${DATA_ARGS[@]}" "${VOCAB_ARGS[@]}" --output-dir "$OUT" --max-samples 64 --grad-accum 4 --max-steps 2 \
      --epochs 1 --eval-limit 8 --log-every 1 --profile-stages "$@" 2>&1 | tee "$OUT/smoke.log" ;;
  lengths)
    "$PY" specdec/dspark/measure_lengths.py --model "${SMOKE_MODEL:-$BASE_MODEL}" "${DATA_ARGS[@]}" \
      --output "$OUT/lengths.json" "$@" 2>&1 | tee "$OUT/lengths.log" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | tee "$OUT/gpu_after.csv"
echo "outputs: $OUT"
