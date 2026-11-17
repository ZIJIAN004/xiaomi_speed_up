#!/usr/bin/env bash
# Diagnose the Diligram Xiaomi-OCR-0 runtime environment.
# This script only reports configuration; it does not install, remove, or modify packages.

set -u

section() {
    printf '\n===== %s =====\n' "$1"
}

run_report() {
    local label="$1"
    shift
    printf '%s: ' "$label"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 || printf '[command failed: exit %s]' "$?"
    else
        printf '[not found]'
    fi
    printf '\n'
}

section "Host"
printf 'hostname: %s\n' "$(hostname 2>/dev/null || printf '[unknown]')"
printf 'kernel: %s\n' "$(uname -a 2>/dev/null || printf '[unknown]')"
printf 'architecture: %s\n' "$(uname -m 2>/dev/null || printf '[unknown]')"

section "Python and Conda"
printf 'CONDA_DEFAULT_ENV: %s\n' "${CONDA_DEFAULT_ENV:-[unset]}"
printf 'CONDA_PREFIX: %s\n' "${CONDA_PREFIX:-[unset]}"
run_report "python" python --version
run_report "python path" command -v python
run_report "pip" python -m pip --version
run_report "uv" uv --version
run_report "uv path" command -v uv

section "CUDA and NVIDIA"
printf 'CUDA_HOME: %s\n' "${CUDA_HOME:-[unset]}"
run_report "nvidia-smi" nvidia-smi
run_report "GPU summary" nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
run_report "nvcc" nvcc --version
if [ -n "${CUDA_HOME:-}" ]; then
    if [ -x "$CUDA_HOME/bin/nvcc" ]; then
        printf 'CUDA_HOME nvcc: %s\n' "$CUDA_HOME/bin/nvcc"
        "$CUDA_HOME/bin/nvcc" --version 2>&1 || true
    else
        printf 'CUDA_HOME nvcc: [not found at %s/bin/nvcc]\n' "$CUDA_HOME"
    fi
fi

section "Package indexes"
python -m pip config list 2>&1 || true
printf 'Relevant environment variables:\n'
env | LC_ALL=C grep -E '^(PIP|UV|INDEX|VLLM|CUDA|HF_HOME)_' | sort || true

section "Installed packages"
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for name in ("vllm", "torch", "numpy", "transformers", "openai", "Pillow", "pypdfium2"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: [not installed]")

try:
    import torch
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
except Exception as exc:
    print(f"torch runtime check: [failed: {exc!r}]")

try:
    import vllm
    print(f"vllm import: OK ({vllm.__version__})")
except Exception as exc:
    print(f"vllm import: [failed: {exc!r}]")
PY

section "uv resolver capabilities"
if command -v uv >/dev/null 2>&1; then
    uv pip --help 2>&1 | grep -E 'index|install|uninstall' || true
    printf '\nuv pip install help (torch backend):\n'
    uv pip install --help 2>&1 | grep -E 'torch-backend|index-url|extra-index-url' || true
else
    printf 'uv: [not found]\n'
fi

section "Hugging Face cache"
printf 'HF_HOME: %s\n' "${HF_HOME:-[unset]}"
for cache in "${HF_HOME:-}" "$HOME/.cache/huggingface"; do
    if [ -n "$cache" ] && [ -d "$cache" ]; then
        printf 'cache: %s\n' "$cache"
        du -sh "$cache" 2>/dev/null || true
        find "$cache" -maxdepth 2 -type d \( -iname '*xiaomi*' -o -iname '*seer*' \) -print 2>/dev/null || true
    fi
done

section "Repository"
printf 'working tree: %s\n' "$(git rev-parse --show-toplevel 2>/dev/null || printf '[not a git repository]')"
git status --short --branch 2>/dev/null || true
git remote -v 2>/dev/null || true

printf '\nDiagnostic completed. No installation or environment changes were made.\n'
