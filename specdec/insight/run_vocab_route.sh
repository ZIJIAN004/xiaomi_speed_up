#!/usr/bin/env bash
set -euo pipefail

# Block-routed draft vocabulary study (inside WSL, minicpm_ocr env). CPU only so far.
#
#   bash specdec/insight/run_vocab_route.sh group-a   # one grouping A (K=$K groups, $SHARED_N shared)
#   bash specdec/insight/run_vocab_route.sh sweep-a   # A for K in $SWEEP_K x shared in $SWEEP_SHARED, then coverage
#   bash specdec/insight/run_vocab_route.sh coverage [groups.pt ...]   # default: every *.pt in $OUT
#
# Outputs: runs/vocab_route/$TASK/ (A_K<K>_S<S>.pt/.json, coverage.json, logs).
# Extra arguments after the mode go to the Python script.
MODE="${1:?usage: run_vocab_route.sh group-a|sweep-a|coverage [extra args]}"; shift
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/mr_jiang/models/minicpm-ocr-1.0}"
BASE_MODEL="${BASE_MODEL:-$MODEL_ROOT/model}"
OFFICIAL_DRAFT="${OFFICIAL_DRAFT:-$MODEL_ROOT/dspark}"
TASK="${TASK:-whole_card}"
case "$TASK" in   # same data and LoRA as specdec/dspark/run_dspark.sh
  whole_card)
    DEFAULT_LORA_DIR=/home/mr_jiang/speed_up/model/MiniCPM-OCR-1.0-LoRA-whole-card_20261126_001627_295140
    DEFAULT_DATA_DIR=/home/mr_jiang/Lora/datasets/IDSpace/whole_card ;;
  kie)
    DEFAULT_LORA_DIR=/home/mr_jiang/speed_up/model/MiniCPM-OCR-1.0-LoRA-20261123_230559_273596-step6754
    DEFAULT_DATA_DIR=/home/mr_jiang/Lora/runs/minicpm_official_9to1_20261123_230559_273596 ;;
  *) echo "TASK must be whole_card or kie, got $TASK" >&2; exit 2 ;;
esac
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DATA_DIR/validation.jsonl}"
# Patched draft rows take the FINE-TUNED target head row (attach_verifier); fall back to the base.
FT_MODEL="${FT_MODEL:-${LORA_DIR:-$DEFAULT_LORA_DIR}/merged}"
if [[ -z "${TARGET_MODEL:-}" ]]; then
  if [[ -d "$FT_MODEL" ]]; then TARGET_MODEL="$FT_MODEL"; else TARGET_MODEL="$BASE_MODEL"; fi
fi
DRAFT_VOCAB="${DRAFT_VOCAB-$PROJECT_ROOT/runs/dspark/${TASK}_draft_vocab/draft_vocab.pt}"   # DRAFT_VOCAB= : official
HEAD_SOURCE="${HEAD_SOURCE:-checkpoint}"
K="${K:-1024}"
SHARED_N="${SHARED_N:-2048}"
SWEEP_K="${SWEEP_K:-256 1024 4096}"
SWEEP_SHARED="${SWEEP_SHARED:-0 2048}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-$PROJECT_ROOT/runs/vocab_route/$TASK}"

VOCAB_ARGS=()
if [[ -n "$DRAFT_VOCAB" ]]; then
  [[ -f "$DRAFT_VOCAB" ]] || { echo "missing $DRAFT_VOCAB (run_dspark.sh vocab), or DRAFT_VOCAB= for official" >&2; exit 2; }
  VOCAB_ARGS=(--draft-vocab "$DRAFT_VOCAB")
fi
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
source "$PROJECT_ROOT/envs/wsl_keepalive.sh"
start_wsl_keepalive "$$" || true
cd "$PROJECT_ROOT"
echo "task=$TASK target=$TARGET_MODEL draft_vocab=${DRAFT_VOCAB:-official} head=$HEAD_SOURCE out=$OUT"

group_a() {   # $1 = groups, $2 = shared, rest = extra args
  local k="$1" s="$2"; shift 2
  "$PY" specdec/insight/group_lm_head.py --official-draft "$OFFICIAL_DRAFT" "${VOCAB_ARGS[@]}" \
    --target-model "$TARGET_MODEL" --head-source "$HEAD_SOURCE" --train-jsonl "$TRAIN_JSONL" \
    --num-groups "$k" --shared "$s" --device "$DEVICE" --output "$OUT/A_K${k}_S${s}" "$@" \
    2>&1 | tee "$OUT/A_K${k}_S${s}.log"
}

coverage() {   # args = groups files (+ extra args)
  "$PY" specdec/insight/block_coverage.py --tokenizer "$TARGET_MODEL" --val-jsonl "$VAL_JSONL" \
    --output "$OUT/coverage.json" --groups "$@" 2>&1 | tee "$OUT/coverage.log"
}

case "$MODE" in
  group-a) group_a "$K" "$SHARED_N" "$@" ;;
  sweep-a)
    for s in $SWEEP_SHARED; do for k in $SWEEP_K; do group_a "$k" "$s" "$@"; done; done
    coverage "$OUT"/A_K*_S*.pt ;;
  coverage)
    if [[ $# -gt 0 ]]; then coverage "$@"; else coverage "$OUT"/*.pt; fi ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "outputs: $OUT"
