#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Results must be new: no historical output reuse.
OMNIDOC_ROOT=${1:?missing OmniDocBench image root}
MIDV500_ROOT=${2:?missing MIDV-500 image root}
MIDV_LAIT_ROOT=${3:?missing MIDV-LAIT image root}
MODEL_PATH=${4:?missing model path}
RESULT_ROOT=${5:?missing result root}
GPU_ID=${GPU_ID:-1}
PORT=${PORT:-26000}
MODEL_NAME=${MODEL_NAME:-Xiaomi-OCR-0}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
DTYPE=${DTYPE:-auto}
WARMUP=${WARMUP:-5}
REPRESENTATIVES=${REPRESENTATIVES:-5}
PROFILE_MAX_ITERATIONS=${PROFILE_MAX_ITERATIONS:-64}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
if [[ -e "$RESULT_ROOT" ]]; then
  echo "Refusing to reuse result directory: $RESULT_ROOT" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT"
RESULT_ROOT=$(cd "$RESULT_ROOT" && pwd)
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$LOG_DIR" "$RESULT_ROOT/profiler_raw" "$RESULT_ROOT/processor_raw"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Required by vLLM versions which gate /collective_rpc and profiler endpoints.
# Bind localhost only: never expose these development endpoints to a network.
export VLLM_SERVER_DEV_MODE=1
export VLLM_TORCH_PROFILER_DIR="$RESULT_ROOT/profiler_raw"

PREP_ARGS=()
[[ -z "${OMNIDOC_ANNOTATIONS:-}" ]] || PREP_ARGS+=(--omnidocbench-annotations "$OMNIDOC_ANNOTATIONS")
[[ -z "${MIDV500_ANNOTATIONS:-}" ]] || PREP_ARGS+=(--midv500-annotations "$MIDV500_ANNOTATIONS")
[[ -z "${MIDV_LAIT_ANNOTATIONS:-}" ]] || PREP_ARGS+=(--midv-lait-annotations "$MIDV_LAIT_ANNOTATIONS")
if [[ "${BENCHMARK_STARTUP_ONLY:-0}" != 1 ]]; then
  python -m benchmark.prepare_datasets --output-dir "$RESULT_ROOT/manifests" \
  --omnidocbench "$OMNIDOC_ROOT" --midv500 "$MIDV500_ROOT" --midv-lait "$MIDV_LAIT_ROOT" "${PREP_ARGS[@]}"
fi

echo "Checking vLLM version..."
vllm --version > "$LOG_DIR/vllm_version.txt" 2>&1
echo "Checking vLLM serve options..."
if ! vllm serve --help=all > "$LOG_DIR/vllm_serve_help.txt" 2>&1; then
  vllm serve --help > "$LOG_DIR/vllm_serve_help.txt" 2>&1
fi
if ! grep -q -- '--worker-extension-cls' "$LOG_DIR/vllm_serve_help.txt"; then
  echo "This vLLM lacks --worker-extension-cls. Supply its version before adapting worker instrumentation; no fake metrics will be produced." >&2
  exit 2
fi
EXTRA_ARGS=(--worker-extension-cls benchmark.worker_extension.BenchmarkWorkerExtension --generation-config vllm)
if [[ "${TRUST_REMOTE_CODE:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--trust-remote-code)
fi
# New versions use a JSON profiler config. Older versions use the documented env interface.
if grep -q -- '--profiler-config' "$LOG_DIR/vllm_serve_help.txt"; then
  PROFILER_CONFIG=$(python - "$RESULT_ROOT/profiler_raw" "$PROFILE_MAX_ITERATIONS" <<'PY'
import json, sys
from vllm.config import ProfilerConfig
fields = set(getattr(ProfilerConfig, "__dataclass_fields__", {})) | set(getattr(ProfilerConfig, "__annotations__", {}))
config = {"profiler": "torch", "torch_profiler_dir": sys.argv[1],
          "torch_profiler_with_stack": True, "torch_profiler_record_shapes": True,
          "torch_profiler_with_memory": True, "torch_profiler_with_flops": True}
if "max_iterations" in fields:
    config["max_iterations"] = int(sys.argv[2])
if "ignore_frontend" in fields:
    config["ignore_frontend"] = True
print(json.dumps(config))
PY
)
  EXTRA_ARGS+=(--profiler-config "$PROFILER_CONFIG")
else
  export VLLM_TORCH_PROFILER_WITH_STACK=1
  export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
  export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
  export VLLM_TORCH_PROFILER_WITH_FLOPS=1
fi
python - "$RESULT_ROOT/server_command.json" "$MODEL_PATH" "$MODEL_NAME" "$PORT" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" "$DTYPE" "${EXTRA_ARGS[@]}" <<'PY'
import json, sys
path, model, name, port, util, length, dtype, *extra = sys.argv[1:]
command = ["vllm", "serve", model, "--served-model-name", name, "--host", "127.0.0.1", "--port", port,
           "--gpu-memory-utilization", util, "--max-model-len", length, "--dtype", dtype, *extra]
with open(path, "w", encoding="utf-8") as f:
    json.dump(command, f, indent=2)
PY
python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError as exc:
        raise SystemExit(f"Benchmark port is unavailable; refusing to attach to an unrelated server: {exc}")
PY
STARTUP_PROBE_DIR=""
READINESS_ARGS=()
if [[ "${BENCHMARK_CAPTURE_STARTUP:-0}" == 1 ]]; then
  STARTUP_PROBE_DIR="$RESULT_ROOT/startup_evidence"
  mkdir -p "$STARTUP_PROBE_DIR"
  READINESS_ARGS+=(--diagnostic-dir "$STARTUP_PROBE_DIR" --server-log "$LOG_DIR/vllm_server.log" --application-ready-grace 30)
fi
echo "Starting vLLM; server log: $LOG_DIR/vllm_server.log"
SERVER_START=$(python -c 'import time; print(time.time())')
VLLM_BENCHMARK_STARTUP_DIR="$STARTUP_PROBE_DIR" \
VLLM_BENCHMARK_PROCESSOR_DIR="$RESULT_ROOT/processor_raw" \
PYTHONPATH="$ROOT_DIR/benchmark/server_bootstrap:$PYTHONPATH" \
vllm serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" --host 127.0.0.1 \
  --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" "${EXTRA_ARGS[@]}" > "$LOG_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!
cleanup() {
  # Only this launcher-recorded PID. No pgrep/pkill or shared-host process matching.
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
python -m benchmark.readiness --base-url "http://127.0.0.1:$PORT" --pid "$VLLM_PID" \
  --launched-at "$SERVER_START" --output "$RESULT_ROOT/server_startup.json" \
  --timeout "${READINESS_TIMEOUT:-1200}" "${READINESS_ARGS[@]}"
if [[ "${BENCHMARK_STARTUP_ONLY:-0}" == 1 ]]; then
  echo "Startup-only check completed: no warmup, inference, processor audit or profiling requests sent."
  exit 0
fi
# Fail early if worker instrumentation is unreachable, rather than running a misleading client-only benchmark.
python - "$PORT" <<'PY'
import sys
from benchmark.monitor import rpc
snapshots = rpc(f"http://127.0.0.1:{sys.argv[1]}", "benchmark_snapshot", ["false"])
if not snapshots or any(s.get("peak_allocated_bytes") is None for s in snapshots):
    raise SystemExit("worker memory RPC unavailable; inspect installed vLLM interface")
print("Worker instrumentation ready")
PY
DATASETS=(omnidocbench midv500 midv-lait)
if [[ "${BENCHMARK_RUNTIME_SMOKE:-0}" == 1 ]]; then
  DATASETS=(omnidocbench)
  python - "$PORT" "$RESULT_ROOT/runtime_interfaces.json" <<'PYCODE'
import json, sys
from benchmark.monitor import rpc
try:
    data = {"status": "observed", "workers": rpc(f"http://127.0.0.1:{sys.argv[1]}", "benchmark_inspect_runtime")}
except Exception as exc:
    data = {"status": "unavailable", "error": str(exc)}
with open(sys.argv[2], "w") as f:
    json.dump(data, f, indent=2)
PYCODE
fi
for DATASET in "${DATASETS[@]}"; do
  RUN_ARGS=(--output-retention "${OUTPUT_RETENTION:-sampled}" --retain-samples "${RETAIN_SAMPLES:-10}"
            --retain-worst "${RETAIN_WORST:-10}" --retention-seed "${RETENTION_SEED:-42}")
  [[ -z "${MAX_TOKENS:-}" ]] || RUN_ARGS+=(--max-tokens "$MAX_TOKENS")
  [[ "${TRUST_REMOTE_CODE:-0}" != 1 ]] || RUN_ARGS+=(--trust-remote-code)
  [[ "${RETRY_REPETITIVE:-1}" != 1 ]] || RUN_ARGS+=(--retry-repetitive)
  [[ -z "${MODULE_MAP:-}" ]] || RUN_ARGS+=(--module-map "$MODULE_MAP")
  [[ -z "${DCGM_URL:-}" ]] || RUN_ARGS+=(--dcgm-url "$DCGM_URL")
  [[ -z "${GPU_COST_PER_HOUR:-}" ]] || RUN_ARGS+=(--gpu-cost-per-hour "$GPU_COST_PER_HOUR")
  case "$DATASET" in
    omnidocbench) QUALITY_COMMAND=${OMNIDOC_QUALITY_COMMAND:-} ;;
    midv500) QUALITY_COMMAND=${MIDV500_QUALITY_COMMAND:-} ;;
    midv-lait) QUALITY_COMMAND=${MIDV_LAIT_QUALITY_COMMAND:-} ;;
  esac
  [[ -z "$QUALITY_COMMAND" ]] || RUN_ARGS+=(--quality-command "$QUALITY_COMMAND")
  python -m benchmark.run_benchmark --backend http --manifest "$RESULT_ROOT/manifests/$DATASET.jsonl" \
    --output "$RESULT_ROOT/$DATASET" --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --base-url "http://127.0.0.1:$PORT" --workers 8 --gpu "$GPU_ID" --server-pid "$VLLM_PID" \
    --request-timeout "$REQUEST_TIMEOUT" --max-model-len "$MAX_MODEL_LEN" \
    --warmup "$WARMUP" --representatives "$REPRESENTATIVES" --profiler-dir "$RESULT_ROOT/profiler_raw" \
    --server-processor-dir "$RESULT_ROOT/processor_raw" \
    --profile-max-iterations "$PROFILE_MAX_ITERATIONS" \
    "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/$DATASET.log"
done
python - "$RESULT_ROOT" "${DATASETS[@]}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reports = {name: json.loads((root/name/"acceptance_report.json").read_text())
           for name in sys.argv[2:]}
(root/"acceptance_report.json").write_text(json.dumps(reports, indent=2) + "\n")
for name, report in reports.items():
    missing = [r["id"] for r in report["criteria"] if r["status"] != "pass"]
    print(name, "PASS" if not missing else "INCOMPLETE: " + ", ".join(missing))
PY
