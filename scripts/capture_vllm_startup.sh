#!/usr/bin/env bash
# Load the model once; capture startup only, never submit inference requests.
set -euo pipefail
MODEL_PATH=${1:?usage: bash scripts/capture_vllm_startup.sh MODEL_PATH [OUTPUT_DIR]}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${2:-$ROOT_DIR/scripts/startup_check_$(date +%Y%m%d_%H%M%S)}
[[ -d "$MODEL_PATH" ]] || { echo "Model directory does not exist" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists; refusing overwrite" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
cd "$ROOT_DIR"
echo "Startup-only evidence capture: loads model on GPU=${GPU_ID:-1}; no inference requests. Output: $OUTPUT_DIR"
status=0
BENCHMARK_STARTUP_ONLY=1 BENCHMARK_CAPTURE_STARTUP=1 READINESS_TIMEOUT="${READINESS_TIMEOUT:-1200}" \
  bash benchmark/run_vllm_concurrency8.sh "$ROOT_DIR/example_pics" "$ROOT_DIR/example_pics" "$ROOT_DIR/example_pics" \
  "$MODEL_PATH" "$OUTPUT_DIR/results" || status=$?
python scripts/summarize_startup_evidence.py --run-root "$OUTPUT_DIR/results" --output "$OUTPUT_DIR/startup_report.json"
echo "Evidence preserved: $OUTPUT_DIR"
exit "$status"
