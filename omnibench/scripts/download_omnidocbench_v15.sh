#!/usr/bin/env bash
set -euo pipefail

# Download the image inputs and GT required for the OmniDocBench v1.5
# two-stage Xiaomi-OCR-0 benchmark. PDFs and demo assets are intentionally
# excluded because the benchmark consumes page images.
DATA_DIR="${1:-/home/mr_jiang/speed_up/Xiaomi-OCR-0/benchmark/data}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DATASET="opendatalab/OmniDocBench"
REVISION="v1_5"
EXPECTED_IMAGES=1355

mkdir -p "$DATA_DIR"

if command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD=(huggingface-cli)
else
    printf '%s\n' "ERROR: neither 'hf' nor 'huggingface-cli' is available." >&2
    printf '%s\n' "Install huggingface_hub in the active environment first." >&2
    exit 1
fi

printf 'Downloading %s revision %s to %s\n' "$DATASET" "$REVISION" "$DATA_DIR"
printf 'HF endpoint: %s\n' "$HF_ENDPOINT"

HF_ENDPOINT="$HF_ENDPOINT" "${HF_CMD[@]}" download "$DATASET" \
    --repo-type dataset \
    --revision "$REVISION" \
    --include 'images/*' \
    --include 'OmniDocBench.json' \
    --local-dir "$DATA_DIR"

GT="$DATA_DIR/OmniDocBench.json"
IMAGE_DIR="$DATA_DIR/images"

if [[ ! -f "$GT" ]]; then
    printf '%s\n' "ERROR: missing $GT" >&2
    exit 2
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
    printf '%s\n' "ERROR: missing $IMAGE_DIR" >&2
    exit 3
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')
printf 'Downloaded image count: %s\n' "$IMAGE_COUNT"
if [[ "$IMAGE_COUNT" -ne "$EXPECTED_IMAGES" ]]; then
    printf 'ERROR: expected %s v1.5 images, found %s\n' "$EXPECTED_IMAGES" "$IMAGE_COUNT" >&2
    exit 4
fi

printf 'Dataset size:\n'
du -sh "$DATA_DIR"
printf '%s\n' "OmniDocBench v1.5 download and validation passed."
printf 'Data directory: %s\n' "$DATA_DIR"
