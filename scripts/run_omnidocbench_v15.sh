#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
# Repository-relative defaults; environment variables can override them.
OMNIDOC_ROOT="${OMNIDOC_ROOT:-$ROOT_DIR/benchmark/data}"
# Fill a local, clean opendatalab/OmniDocBench checkout on branch v1_5.
OMNIDOC_EVALUATOR_ROOT="${OMNIDOC_EVALUATOR_ROOT:-$ROOT_DIR/evaluation_pipeline/OmniDocBench}"
OMNIDOC_EVAL_BACKEND="${OMNIDOC_EVAL_BACKEND:-docker}"
OMNIDOC_EVAL_IMAGE="${OMNIDOC_EVAL_IMAGE:-sunyuefeng/omnidocbench-env:v1.5}"
OMNIDOC_EVAL_PYTHON="${OMNIDOC_EVAL_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-/home/mr_jiang/models/xiaomi-ocr-0/model}"
GPU_ID="${GPU_ID:-1}"
[[ -d "$OMNIDOC_ROOT/images" && -f "$OMNIDOC_ROOT/OmniDocBench.json" ]] || { echo "Missing images/ or OmniDocBench.json under $OMNIDOC_ROOT" >&2; exit 2; }
[[ -f "$OMNIDOC_EVALUATOR_ROOT/pdf_validation.py" ]] || { echo "Missing v1_5 evaluator: $OMNIDOC_EVALUATOR_ROOT/pdf_validation.py (the dataset does not include evaluator code)." >&2; exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one physical GPU index.' >&2; exit 2; }
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/benchmark/results/omnidocbench_v15_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$RESULT_ROOT" && ! -e "${RESULT_ROOT}.log" ]] || { echo 'Result path already exists; refusing reuse.' >&2; exit 2; }
export GPU_ID CUDA_VISIBLE_DEVICES="$GPU_ID" TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT_DIR/benchmark/server_bootstrap:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMNIDOCBENCH_PYTHON_COMPAT="$ROOT_DIR/benchmark/skimage_compat.py"
mkdir -p "$(dirname "$RESULT_ROOT")"
ARGS=(--dataset "$OMNIDOC_ROOT" --model "$MODEL_PATH" --output "$RESULT_ROOT" --gpu "$GPU_ID"
      --evaluator-root "$OMNIDOC_EVALUATOR_ROOT" --eval-python "$OMNIDOC_EVAL_PYTHON"
      --eval-backend "$OMNIDOC_EVAL_BACKEND" --eval-image "$OMNIDOC_EVAL_IMAGE"
      --warmup "${WARMUP:-5}" --representatives "${REPRESENTATIVES:-5}"
      --profile-max-iterations "${PROFILE_MAX_ITERATIONS:-64}"
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" --max-model-len "${MAX_MODEL_LEN:-16384}"
      --dtype "${DTYPE:-auto}" --monitor-interval "${MONITOR_INTERVAL:-1}"
      --request-timeout "${REQUEST_TIMEOUT:-600}"
      --output-retention "${OUTPUT_RETENTION:-sampled}" --retain-samples "${RETAIN_SAMPLES:-10}"
      --retain-worst "${RETAIN_WORST:-10}" --retention-seed "${RETENTION_SEED:-42}")
[[ -z "${MAX_TOKENS:-}" ]] || ARGS+=(--max-tokens "$MAX_TOKENS")
[[ -z "${DCGM_URL:-}" ]] || ARGS+=(--dcgm-url "$DCGM_URL")
[[ "${RETRY_REPETITIVE:-1}" != 1 ]] || ARGS+=(--retry-repetitive)
echo 'OmniDocBench 1.5: offline whole pages(workers=8; no PaddleX) -> page evaluation'
python -m benchmark.run_omnidocbench_v15 "${ARGS[@]}" 2>&1 | tee "${RESULT_ROOT}.log"
