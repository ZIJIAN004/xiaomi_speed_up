#!/usr/bin/env bash
# Diagnostic only: loads a real model; short output is not a performance baseline.
set -euo pipefail
MODEL_PATH=${1:?usage: bash baseline/scripts/run_benchmark_runtime_check.sh MODEL_PATH [IMAGE_PATH] [OUTPUT_DIR]}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE_PATH=${2:-$ROOT_DIR/example_pics/receipt.jpg}
OUTPUT_DIR=${3:-$ROOT_DIR/runs/runtime_check_$(date +%Y%m%d_%H%M%S)}
[[ -d "$MODEL_PATH" ]] || { echo "Model directory does not exist" >&2; exit 2; }
[[ -f "$IMAGE_PATH" ]] || { echo "Image does not exist" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "Output already exists; refusing overwrite" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR/input"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
cd "$ROOT_DIR"
echo "Offline runtime diagnostic: GPU=${GPU_ID:-1}, model will load; 8 copies of one image, max_tokens=32. Not a quality/performance baseline."
python baseline/scripts/check_benchmark_interfaces.py --deep --model-path "$MODEL_PATH" --output "$OUTPUT_DIR/interfaces_deep.json"
python - "$IMAGE_PATH" "$OUTPUT_DIR/input" <<'PY'
from pathlib import Path
import shutil, sys
source, directory = Path(sys.argv[1]), Path(sys.argv[2])
for index in range(8):
    shutil.copy2(source, directory / (f"smoke_{index:02d}" + source.suffix))
PY
status=0
OMNIDOC_QUALITY_COMMAND= BENCHMARK_RUNTIME_SMOKE=1 MAX_TOKENS=32 WARMUP=3 REPRESENTATIVES=3 RETRY_REPETITIVE=0 PROFILE_MAX_ITERATIONS=16 \
  bash baseline/harness/run_offline_concurrency8.sh "$OUTPUT_DIR/input" "$OUTPUT_DIR/input" "$OUTPUT_DIR/input" \
  "$MODEL_PATH" "$OUTPUT_DIR/results" || status=$?
python baseline/scripts/summarize_benchmark_runtime.py --run-root "$OUTPUT_DIR/results" --output "$OUTPUT_DIR/runtime_verification.json" || status=$?
echo "Runtime diagnostic saved under: $OUTPUT_DIR"
exit "$status"
