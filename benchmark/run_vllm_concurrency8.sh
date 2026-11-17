#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./benchmark/run_vllm_concurrency8.sh \
#     /path/to/omnidocbench /path/to/midv500 /path/to/midv-lait \
#     /path/to/model /path/to/results

OMNIDOC_ROOT=${1:?missing OmniDocBench image root}
MIDV500_ROOT=${2:?missing MIDV-500 image root}
MIDV_LAIT_ROOT=${3:?missing MIDV-LAIT image root}
MODEL_PATH=${4:?missing model path}
RESULT_ROOT=${5:?missing result root}

GPU_ID=${GPU_ID:-1}
PORT=${PORT:-26000}
MODEL_NAME=${MODEL_NAME:-Xiaomi-OCR-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.80}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
MAX_TOKENS=${MAX_TOKENS:-16384}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST_DIR="$RESULT_ROOT/manifests"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$RESULT_ROOT" "$LOG_DIR"

python "$ROOT_DIR/benchmark/prepare_datasets.py" \
  --output-dir "$MANIFEST_DIR" \
  --omnidocbench "$OMNIDOC_ROOT" \
  --midv500 "$MIDV500_ROOT" \
  --midv-lait "$MIDV_LAIT_ROOT"

# Start one vLLM server on physical GPU 1. No --enforce-eager: keep vLLM defaults.
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len 16384 \
  > "$LOG_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT

python - "$PORT" <<'PY'
import sys, time, urllib.request
url = f"http://127.0.0.1:{sys.argv[1]}/v1/models"
for _ in range(240):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print("vLLM server is ready")
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
raise SystemExit("vLLM server did not become ready; inspect logs/vllm_server.log")
PY

declare -A DATASET_START
for DATASET in omnidocbench midv500 midv-lait; do
  DATASET_START[$DATASET]=$(date +%s)
  python -m evaluation_pipeline.e2e_img2md \
    --input-jsonl "$MANIFEST_DIR/$DATASET.jsonl" \
    --output_dir "$RESULT_ROOT/predictions/$DATASET" \
    --model_path "$MODEL_PATH" \
    --served_model_name "$MODEL_NAME" \
    --task document \
    --gpu_devices "$GPU_ID" \
    --port "$PORT" \
    --workers 8 \
    --max_tokens "$MAX_TOKENS" \
    --request_timeout "$REQUEST_TIMEOUT" \
    --presence_penalty 0.0 \
    --repetition_penalty 1.0 \
    --retry-repetitive \
    --log_dir "$LOG_DIR/$DATASET"
done

for DATASET in omnidocbench midv500 midv-lait; do
  printf '%s\n' "$(( $(date +%s) - DATASET_START[$DATASET] ))" > "$RESULT_ROOT/$DATASET.elapsed_s"
done

python "$ROOT_DIR/benchmark/collect_results.py" \
  --result-root "$RESULT_ROOT" \
  --model "$MODEL_NAME" \
  --gpu "$GPU_ID" \
  --concurrency 8 \
  --batch-size 16
