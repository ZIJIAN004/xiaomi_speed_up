#!/usr/bin/env bash
set -euo pipefail
OMNIDOC_ROOT=${1:?missing OmniDocBench 1.5 image root}
MIDV500_ROOT=${2:?missing MIDV-500 image root}
MIDV_LAIT_ROOT=${3:?missing MIDV-LAIT image root}
MODEL_PATH=${4:?missing local model directory}
RESULT_ROOT=${5:?missing new output directory}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
GPU_ID=${GPU_ID:-1}
# Exactly one physical GPU. Never accept a comma-separated GPU list.
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be one GPU index" >&2; exit 2; }
[[ ! -e "$RESULT_ROOT" && ! -e "${RESULT_ROOT}.log" ]] || { echo "Output already exists; refusing reuse" >&2; exit 2; }
export GPU_ID CUDA_VISIBLE_DEVICES="$GPU_ID" TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$ROOT_DIR/benchmark/server_bootstrap:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "Single-GPU offline benchmark: GPU=$GPU_ID, max in-flight=8; no HTTP server or readiness wait."
mkdir -p "$(dirname "$RESULT_ROOT")"
python -m benchmark.run_offline "$OMNIDOC_ROOT" "$MIDV500_ROOT" "$MIDV_LAIT_ROOT" "$MODEL_PATH" "$RESULT_ROOT" 2>&1 | tee "${RESULT_ROOT}.log"
