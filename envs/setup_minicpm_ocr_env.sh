#!/usr/bin/env bash
set -euo pipefail

# Create the MiniCPM-OCR 1.0 environment (vLLM inference incl. DSpark + DSpark draft training),
# isolated from the Xiaomi-OCR env, and download the main model and the official DSpark draft.
#
#   vLLM 0.34.0 PyPI wheel: registers MiniCPMOcrForConditionalGeneration (SupportsEagle3 aux
#   hidden states) and Qwen3DSparkModel; pins torch==2.14.0, transformers>=5.16.1,<5.23.
#   MiniCPM-OCR needs transformers>=5.21,<5.23 -> 5.22.x.
#   speculators (DSpark model/loss) pins transformers<=5.19 -> installed with --no-deps.
#
# Re-runnable: an existing env is reused and finished downloads are skipped.
# Usage: bash envs/setup_minicpm_ocr_env.sh   (inside WSL; keeps the distro alive itself)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-minicpm_ocr}"
CONDA="${CONDA:-/home/mr_jiang/miniforge3/bin/conda}"
PREFIX="$(dirname "$(dirname "$CONDA")")/envs/$ENV_NAME"
PY="$PREFIX/bin/python"
MODEL_ROOT="${MODEL_ROOT:-/home/mr_jiang/models/minicpm-ocr-1.0}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.34.0}"   # may point at an already downloaded wheel
SPECULATORS_COMMIT="8f993e3243cdfe33f6e124c1e35744d590691cd8"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/runs/setup_${ENV_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG_DIR/setup.log"; }

source "$PROJECT_ROOT/envs/wsl_keepalive.sh"
log "$(start_wsl_keepalive "$$" 2>&1)"
df -h "$(dirname "$PREFIX")" "$(dirname "$MODEL_ROOT")" | tee -a "$LOG_DIR/setup.log"

if [[ ! -x "$PY" ]]; then
  log "Creating conda env $ENV_NAME (python 3.12) at $PREFIX"
  "$CONDA" create -y -n "$ENV_NAME" python=3.12 >>"$LOG_DIR/conda.log" 2>&1
else
  log "Reusing existing env $PREFIX"
fi
pip_install() { log "pip install $*"; "$PY" -m pip install "$@" >>"$LOG_DIR/pip.log" 2>&1; }

pip_install "$VLLM_SPEC"
pip_install "transformers>=5.22.0,<5.23.0"
# Benchmark harness / tooling extras not pulled in by vLLM.
pip_install psutil pyyaml modelscope
# speculators: DSpark draft model + loss only; its transformers<=5.19 pin conflicts with MiniCPM.
pip_install --no-deps "git+https://github.com/vllm-project/speculators.git@$SPECULATORS_COMMIT"
pip_install pydantic pydantic-settings loguru rich typer click safetensors
# Qwen3.5-style linear-attention kernels for HF forward during draft training (fla is Triton-only).
pip_install "flash-linear-attention"
# causal-conv1d compiles CUDA code when no matching wheel exists and there is no system nvcc here;
# it only speeds up HF training forward, so a failure is logged and does not stop the setup.
if ! pip_install causal-conv1d; then
  log "WARNING: causal-conv1d not installed (see pip.log); HF training forward falls back to torch conv"
fi

log "Version / import checks"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG_DIR/setup.log"
import importlib, torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| vllm", vllm.__version__)
assert torch.__version__.startswith("2.14.0"), torch.__version__
assert transformers.__version__.startswith("5.22."), transformers.__version__
assert vllm.__version__ == "0.34.0", vllm.__version__
from transformers import MiniCPMOcrForConditionalGeneration  # noqa: F401  (model_type minicpm_ocr)
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
assert "MiniCPMOcrForConditionalGeneration" in archs and "Qwen3DSparkModel" in archs
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig  # noqa: F401
for name in ("fla", "causal_conv1d", "prometheus_client", "PIL", "numpy", "psutil", "modelscope"):
    try:
        importlib.import_module(name); print("import ok:", name)
    except ImportError as exc:
        print("import MISSING:", name, exc)
PY
"$PY" -m pip check 2>&1 | tee -a "$LOG_DIR/setup.log" || true   # expected: speculators wants transformers<=5.19
(cd "$PROJECT_ROOT" && "$PY" -c "import baseline.harness.run_benchmark, baseline.harness.collect_results, baseline.harness.worker_extension, baseline.omnidocbench.run_omnidocbench_v15; print('harness imports ok')") 2>&1 | tee -a "$LOG_DIR/setup.log"

download() {  # modelscope repo -> local dir, skipped when config.json is already there
  if [[ -f "$2/config.json" ]]; then log "Already downloaded: $2"; return; fi
  log "Downloading $1 -> $2"
  "$PY" -m modelscope.cli.cli download --model "$1" --local_dir "$2" >>"$LOG_DIR/download.log" 2>&1
}
mkdir -p "$MODEL_ROOT"
download OpenBMB/MiniCPM-OCR-1.0 "$MODEL_ROOT/model"
download OpenBMB/MiniCPM-OCR-1.0-DSpark "$MODEL_ROOT/dspark"
"$PY" - "$MODEL_ROOT" <<'PY' 2>&1 | tee -a "$LOG_DIR/setup.log"
import json, sys
root = sys.argv[1]
m = json.load(open(f"{root}/model/config.json")); d = json.load(open(f"{root}/dspark/config.json"))
print("model:", m.get("architectures"), m.get("model_type"))
print("dspark:", d.get("architectures"), "block_size", d.get("block_size"),
      "target_layer_ids", (d.get("dflash_config") or {}).get("target_layer_ids"))
PY
du -sh "$PREFIX" "$MODEL_ROOT"/* | tee -a "$LOG_DIR/setup.log"
log "Setup finished: env $PREFIX, models $MODEL_ROOT"
