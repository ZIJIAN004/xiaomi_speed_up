#!/usr/bin/env bash
set -euo pipefail

# Re-profile only the representative pages of an earlier OmniDocBench run (no evaluator).
# The sub-manifest holds exactly those pages and --representatives equals their count,
# so the same pages are profiled again with the same inference settings.
# Waits until GPU_ID has more than FREE_THRESHOLD_MIB free so it can queue behind another job.
#
# Usage: SOURCE_RUN=<.../omnidocbench_v15 result dir> bash scripts/run_representative_profile.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="${SOURCE_RUN:?set SOURCE_RUN to an omnidocbench_v15 result directory}"
SPEED_UP_PYTHON="${SPEED_UP_PYTHON:-/home/mr_jiang/miniforge3/envs/speed_up/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/mr_jiang/models/xiaomi-ocr-0/model}"
GPU_ID="${GPU_ID:-1}"
FREE_THRESHOLD_MIB="${FREE_THRESHOLD_MIB:-24000}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/benchmark/results/rep_profile_$(date +%Y%m%d_%H%M%S)}"

SELECTION="$SOURCE_RUN/inference/representative_selection.json"
[[ -f "$SELECTION" && -f "$SOURCE_RUN/page_manifest.jsonl" ]] || { echo "Missing $SELECTION or page_manifest.jsonl" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" ]] || { echo "Result path already exists; refusing reuse." >&2; exit 2; }
mkdir -p "$RESULT_ROOT"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$RESULT_ROOT/orchestrator.log"; }

source "$PROJECT_ROOT/scripts/wsl_keepalive.sh"
log "$(start_wsl_keepalive "$$" 2>&1)"

COUNT="$("$SPEED_UP_PYTHON" - "$SELECTION" "$SOURCE_RUN/page_manifest.jsonl" "$RESULT_ROOT/page_manifest.jsonl" <<'PY'
import json, sys
selection, manifest, output = sys.argv[1:]
wanted = {s["sample_id"] for s in json.load(open(selection))["samples"]}
rows = [line for line in open(manifest) if line.strip() and json.loads(line)["sample_id"] in wanted]
if len(rows) != len(wanted):
    sys.exit(f"selection has {len(wanted)} pages but only {len(rows)} found in manifest")
open(output, "w").writelines(rows)
print(len(rows))
PY
)"
log "Sub-manifest: $COUNT representative pages from $SOURCE_RUN"

log "Waiting for GPU $GPU_ID free memory > ${FREE_THRESHOLD_MIB} MiB, checking every ${INTERVAL_SECONDS}s"
while :; do
  free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' | head -n 1)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib > FREE_THRESHOLD_MIB )); then
    log "GPU $GPU_ID free memory: ${free_mib} MiB; starting"
    break
  fi
  sleep "$INTERVAL_SECONDS"
done

# Same runtime environment as omnibench/scripts/run_omnidocbench_v15.sh.
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID" TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$PROJECT_ROOT/benchmark/server_bootstrap:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
status=0
"$SPEED_UP_PYTHON" -m benchmark.run_benchmark --backend offline \
  --manifest "$RESULT_ROOT/page_manifest.jsonl" --output "$RESULT_ROOT/inference" \
  --model "$MODEL_PATH" --gpu "$GPU_ID" --workers 8 \
  --warmup "${WARMUP:-5}" --representatives "$COUNT" \
  --profile-max-iterations "${PROFILE_MAX_ITERATIONS:-64}" --retain-raw-traces "${RETAIN_RAW_TRACES:-5}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --dtype "${DTYPE:-auto}" --monitor-interval "${MONITOR_INTERVAL:-1}" \
  --request-timeout "${REQUEST_TIMEOUT:-600}" --warmup-request-timeout "${WARMUP_REQUEST_TIMEOUT:-1800}" \
  --profiler-dir "$RESULT_ROOT/profiler_raw" --server-processor-dir "$RESULT_ROOT/processor_raw" \
  --output-retention all ${SPECULATIVE_CONFIG:+--speculative-config "$SPECULATIVE_CONFIG"} \
  > "$RESULT_ROOT/run.log" 2>&1 || status=$?
log "Representative profile finished with status $status"
exit "$status"
