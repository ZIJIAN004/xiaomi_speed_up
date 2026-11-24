#!/usr/bin/env bash
set -u

# MiniCPM-OCR 1.0, same tasks and settings as the Xiaomi-OCR-0 baseline:
#   A)  OmniDocBench v1.5, plain decoding     B)  OmniDocBench v1.5, DSpark (official draft, k=15)
#   IA) IDSpace validation, plain decoding    IB) IDSpace validation, DSpark
# Runs the selected items serially on one GPU, in the order given. Before EACH run it waits until the GPU has
# more than FREE_THRESHOLD_MIB free (default 22 GiB). A failure in one run is recorded and does not stop the rest.
# Inference runs in the minicpm_ocr env (routed by model_type); OmniDocBench scoring in the omnidocbench_v15 env.
#
# Usage (inside WSL, ideally in tmux):  RUNS="A B IA IB" bash baseline/scripts/run_minicpm_omnidocbench.sh
#   Default RUNS="A" (DSpark off). NUM_SPECULATIVE_TOKENS changes k for B / IB.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/mr_jiang/models/minicpm-ocr-1.0/model}"
DRAFT_PATH="${DRAFT_PATH:-/home/mr_jiang/models/minicpm-ocr-1.0/dspark}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"  # = draft block_size
MINICPM_PYTHON="${MINICPM_PYTHON:-/home/mr_jiang/miniforge3/envs/minicpm_ocr/bin/python}"
OMNIDOC_EVAL_PYTHON="${OMNIDOC_EVAL_PYTHON:-/home/mr_jiang/miniforge3/envs/omnidocbench_v15/bin/python}"
GPU_ID="${GPU_ID:-1}"
FREE_THRESHOLD_MIB="${FREE_THRESHOLD_MIB:-22528}"  # 22 GiB
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
RUNS="${RUNS:-A}"  # default: plain decoding only; RUNS="A B" or "B" adds DSpark
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/minicpm_serial_$(date +%Y%m%d_%H%M%S)}"
OMNIDOC_ROOT="${OMNIDOC_ROOT:-$PROJECT_ROOT/data}"
OMNIDOC_EVALUATOR_ROOT="${OMNIDOC_EVALUATOR_ROOT:-$PROJECT_ROOT/evaluation_pipeline/OmniDocBench}"
IDSPACE_JSONL="${IDSPACE_JSONL:-/home/mr_jiang/Lora/runs/idspace_e3/prepared/validation.jsonl}"
IDSPACE_IMAGE_ROOT="${IDSPACE_IMAGE_ROOT:-/home/mr_jiang/Lora/runs/idspace_e3/prepared}"  # JSONL image paths start with images/
SPEC_CONFIG="{\"method\":\"dspark\",\"model\":\"$DRAFT_PATH\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"
LOG_ROOT="$RUN_ROOT/logs"
[[ ! -e "$RUN_ROOT/status.json" ]] || { echo "Run root already used: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$LOG_ROOT"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/orchestrator.log"; }

# WSL powers off when the last wsl.exe client exits, killing tmux with it. Hold the distro until this script exits.
source "$PROJECT_ROOT/envs/wsl_keepalive.sh"
log "$(start_wsl_keepalive "$$" 2>&1)"

check() {  # check <test-expression-result> <description>
  if [[ "$1" == ok ]]; then log "found: $2"; else log "MISSING: $2"; failed=1; fi
}
failed=0
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors" ]] && r=ok || r=no; check "$r" "model $MODEL_PATH"
[[ -f "$DRAFT_PATH/config.json" && -f "$DRAFT_PATH/model.safetensors" ]] && r=ok || r=no; check "$r" "DSpark draft $DRAFT_PATH"
[[ -x "$MINICPM_PYTHON" ]] && r=ok || r=no; check "$r" "inference python $MINICPM_PYTHON"
[[ -x "$OMNIDOC_EVAL_PYTHON" ]] && r=ok || r=no; check "$r" "evaluator python $OMNIDOC_EVAL_PYTHON"
[[ -d "$OMNIDOC_ROOT/images" && -f "$OMNIDOC_ROOT/OmniDocBench.json" ]] && r=ok || r=no; check "$r" "OmniDocBench data $OMNIDOC_ROOT"
[[ -f "$OMNIDOC_EVALUATOR_ROOT/pdf_validation.py" ]] && r=ok || r=no; check "$r" "evaluator $OMNIDOC_EVALUATOR_ROOT"
if [[ " $RUNS " == *" IA "* || " $RUNS " == *" IB "* ]]; then
  [[ -f "$IDSPACE_JSONL" && -d "$IDSPACE_IMAGE_ROOT/images" ]] && r=ok || r=no; check "$r" "IDSpace data $IDSPACE_JSONL"
fi
model_type=$(cd "$PROJECT_ROOT" && python3 -m baseline.harness.model_profiles "$MODEL_PATH" model_type 2>/dev/null)
[[ "$model_type" == minicpm_ocr ]] && r=ok || r=no; check "$r" "model_type=minicpm_ocr (got '${model_type}')"
if (( failed )); then
  log "Validation failed; nothing started"
  exit 2
fi
log "Plan: runs [$RUNS] on GPU $GPU_ID; B speculative_config=$SPEC_CONFIG"

wait_for_gpu() {
  log "Waiting for GPU $GPU_ID free memory > ${FREE_THRESHOLD_MIB} MiB (check every ${INTERVAL_SECONDS}s)"
  while :; do
    free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' | head -n 1)"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib > FREE_THRESHOLD_MIB )); then
      log "GPU $GPU_ID free memory ${free_mib} MiB; starting"
      return
    fi
    sleep "$INTERVAL_SECONDS"
  done
}

run_one() {  # run_one <name> <speculative config or empty>
  local name="$1" spec="$2" status=0
  wait_for_gpu
  log "Starting run $name -> $RUN_ROOT/$name"
  (
    cd "$PROJECT_ROOT" || exit 1
    MODEL_PATH="$MODEL_PATH" \
    MODEL_PYTHON="$MINICPM_PYTHON" \
    SPECULATIVE_CONFIG="$spec" \
    GPU_ID="$GPU_ID" \
    RESULT_ROOT="$RUN_ROOT/$name" \
    OMNIDOC_ROOT="$OMNIDOC_ROOT" \
    OMNIDOC_EVALUATOR_ROOT="$OMNIDOC_EVALUATOR_ROOT" \
    OMNIDOC_EVAL_BACKEND="python" \
    OMNIDOC_EVAL_PYTHON="$OMNIDOC_EVAL_PYTHON" \
    OUTPUT_RETENTION="${OUTPUT_RETENTION:-all}" \
    bash baseline/omnidocbench/scripts/run_omnidocbench_v15.sh
  ) >"$LOG_ROOT/$name.log" 2>&1 || status=$?
  log "Run $name finished with status $status"
  return "$status"
}

run_idspace() {  # run_idspace <name> <speculative config or empty>
  local name="$1" spec="$2" status=0
  wait_for_gpu
  log "Starting run $name -> $RUN_ROOT/$name"
  (
    cd "$PROJECT_ROOT" || exit 1
    spec_args=()
    [[ -n "$spec" ]] && spec_args=(--speculative-config "$spec")
    PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"     "$MINICPM_PYTHON" baseline/idspace/scripts/evaluate_vllm.py       --input-jsonl "$IDSPACE_JSONL" --image-root "$IDSPACE_IMAGE_ROOT"       --model "$MODEL_PATH" --output "$RUN_ROOT/$name" --split validation       --workers 8 --gpu "$GPU_ID" --warmup "${IDSPACE_WARMUP:-5}"       --representatives "${IDSPACE_REPRESENTATIVES:-50}" --output-retention all "${spec_args[@]}"
  ) >"$LOG_ROOT/$name.log" 2>&1 || status=$?
  log "Run $name finished with status $status"
  return "$status"
}

declare -A status=()
for run in $RUNS; do
  case "$run" in
    A) run_one A_plain "" && status[A_plain]=0 || status[A_plain]=$? ;;
    B) run_one B_dspark_k${NUM_SPECULATIVE_TOKENS} "$SPEC_CONFIG" && status[B_dspark_k${NUM_SPECULATIVE_TOKENS}]=0 \
         || status[B_dspark_k${NUM_SPECULATIVE_TOKENS}]=$? ;;
    IA) run_idspace IA_idspace_plain "" && status[IA_idspace_plain]=0 || status[IA_idspace_plain]=$? ;;
    IB) run_idspace IB_idspace_dspark_k${NUM_SPECULATIVE_TOKENS} "$SPEC_CONFIG"           && status[IB_idspace_dspark_k${NUM_SPECULATIVE_TOKENS}]=0 || status[IB_idspace_dspark_k${NUM_SPECULATIVE_TOKENS}]=$? ;;
    *) log "Unknown run '$run' (use A, B, IA, IB)"; status[$run]=2 ;;
  esac
done

overall=0
json="{\"gpu\":$GPU_ID,\"free_threshold_mib\":$FREE_THRESHOLD_MIB,\"model\":\"$MODEL_PATH\",\"draft\":\"$DRAFT_PATH\",\"runs\":{"
sep=""
for name in "${!status[@]}"; do
  json+="$sep\"$name\":${status[$name]}"; sep=","
  (( status[$name] == 0 )) || overall=1
done
printf '%s}}\n' "$json" | tee "$RUN_ROOT/status.json"
exit "$overall"
