#!/usr/bin/env bash
set -u

# Wait for the selected GPU (default GPU 0) to have >22 GiB free, then run both evaluations sequentially.
# A failure in one benchmark is recorded but does not prevent the other.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEED_UP_PYTHON="${SPEED_UP_PYTHON:-/home/mr_jiang/miniforge3/envs/speed_up/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/mr_jiang/models/xiaomi-ocr-0/model}"
GPU_ID="${GPU_ID:-0}"
FREE_THRESHOLD_MIB="${FREE_THRESHOLD_MIB:-22528}"  # 22 GiB
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
RUN_ROOT="${RUN_ROOT:-/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/results/gpu0_serial_$(date +%Y%m%d_%H%M%S)}"
OMNI_OUTPUT="${OMNI_OUTPUT:-$RUN_ROOT/omnidocbench_v15}"
IDSPACE_OUTPUT="${IDSPACE_OUTPUT:-$RUN_ROOT/idspace_validation}"
OMNIDOC_ROOT="${OMNIDOC_ROOT:-$PROJECT_ROOT/benchmark/data}"
OMNIDOC_EVALUATOR_ROOT="${OMNIDOC_EVALUATOR_ROOT:-$PROJECT_ROOT/evaluation_pipeline/OmniDocBench}"
IDSPACE_JSONL="${IDSPACE_JSONL:-/home/mr_jiang/qkr/runs/idspace_e3/prepared/validation.jsonl}"
IDSPACE_IMAGE_ROOT="${IDSPACE_IMAGE_ROOT:-/home/mr_jiang/qkr/runs/idspace_e3/prepared/images}"
LOG_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOG_ROOT"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/orchestrator.log"
}

query_free_mib() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]' | head -n 1
}

validate_paths() {
  local failed=0
  if [[ ! -d "$PROJECT_ROOT" ]]; then
    log "Missing project root: $PROJECT_ROOT"
    failed=1
  fi
  if [[ ! -d "$OMNIDOC_ROOT/images" || ! -f "$OMNIDOC_ROOT/OmniDocBench.json" ]]; then
    log "Missing OmniDocBench data: expected $OMNIDOC_ROOT/images and $OMNIDOC_ROOT/OmniDocBench.json"
    failed=1
  else
    log "OmniDocBench data found: $OMNIDOC_ROOT"
  fi
  if [[ ! -f "$OMNIDOC_EVALUATOR_ROOT/pdf_validation.py" ]]; then
    log "Missing OmniDocBench evaluator: $OMNIDOC_EVALUATOR_ROOT/pdf_validation.py"
    failed=1
  else
    log "OmniDocBench evaluator found: $OMNIDOC_EVALUATOR_ROOT"
  fi
  if [[ ! -f "$IDSPACE_JSONL" ]]; then
    log "Missing IDSpace JSONL: $IDSPACE_JSONL"
    failed=1
  else
    log "IDSpace JSONL found: $IDSPACE_JSONL"
  fi
  if [[ ! -d "$IDSPACE_IMAGE_ROOT" ]]; then
    log "Missing IDSpace image root: $IDSPACE_IMAGE_ROOT"
    failed=1
  else
    log "IDSpace image root found: $IDSPACE_IMAGE_ROOT"
  fi
  if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    log "Missing model config: $MODEL_PATH/config.json"
    failed=1
  else
    log "Model path found: $MODEL_PATH"
  fi
  if [[ ! -x "$SPEED_UP_PYTHON" ]]; then
    log "Missing executable Python: $SPEED_UP_PYTHON"
    failed=1
  else
    log "speed_up Python found: $SPEED_UP_PYTHON"
  fi
  return "$failed"
}

if ! validate_paths; then
  log "Path validation failed; no benchmark will start"
  exit 2
fi

log "Waiting for GPU $GPU_ID free memory > ${FREE_THRESHOLD_MIB} MiB (22 GiB), checking every ${INTERVAL_SECONDS}s"
while :; do
  free_mib="$(query_free_mib)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
    log "GPU $GPU_ID free memory: ${free_mib} MiB"
    if (( free_mib > FREE_THRESHOLD_MIB )); then
      break
    fi
  else
    log "Unable to read GPU $GPU_ID free memory; retrying"
  fi
  sleep "$INTERVAL_SECONDS"
done

export SPEED_UP_PYTHON
export MODEL_PATH
export GPU_ID

omni_status=0
idspace_status=0

log "Starting OmniDocBench v1.5"
(
  cd "$PROJECT_ROOT" || exit 1
  SPEED_UP_PYTHON="$SPEED_UP_PYTHON" \
  MODEL_PATH="$MODEL_PATH" \
  GPU_ID="$GPU_ID" \
  RESULT_ROOT="$OMNI_OUTPUT" \
  OMNIDOC_ROOT="$OMNIDOC_ROOT" \
  OMNIDOC_EVALUATOR_ROOT="$OMNIDOC_EVALUATOR_ROOT" \
  OMNIDOC_EVAL_BACKEND="python" \
  OMNIDOC_EVAL_PYTHON="${OMNIDOC_EVAL_PYTHON:-/home/mr_jiang/miniforge3/envs/omnidocbench_v15/bin/python}" \
  OUTPUT_RETENTION="${OUTPUT_RETENTION:-all}" \
  bash omnibench/scripts/run_omnidocbench_v15.sh
) >"$LOG_ROOT/omnidocbench.log" 2>&1 || omni_status=$?
log "OmniDocBench finished with status $omni_status; continuing regardless"

log "Starting IDSpace validation"
(
  cd "$PROJECT_ROOT" || exit 1
  "$SPEED_UP_PYTHON" idspace/scripts/evaluate_vllm.py \
    --input-jsonl "$IDSPACE_JSONL" \
    --image-root "$IDSPACE_IMAGE_ROOT" \
    --model "$MODEL_PATH" \
    --output "$IDSPACE_OUTPUT" \
    --split validation \
    --workers 8 \
    --gpu "$GPU_ID" \
    --max-tokens "${IDSPACE_MAX_TOKENS:-2048}" \
    --warmup "${IDSPACE_WARMUP:-5}" \
    --representatives "${IDSPACE_REPRESENTATIVES:-5}" \
    --output-retention all
) >"$LOG_ROOT/idspace_validation.log" 2>&1 || idspace_status=$?
log "IDSpace validation finished with status $idspace_status"

printf '{"gpu":%s,"free_threshold_mib":%s,"omnidocbench_status":%s,"idspace_status":%s,"omnidocbench_output":"%s","idspace_output":"%s"}\n' \
  "$GPU_ID" "$FREE_THRESHOLD_MIB" "$omni_status" "$idspace_status" "$OMNI_OUTPUT" "$IDSPACE_OUTPUT" \
  | tee "$RUN_ROOT/status.json"

# Return failure only after both benchmarks have had a chance to run.
if (( omni_status != 0 || idspace_status != 0 )); then
  exit 1
fi
exit 0
