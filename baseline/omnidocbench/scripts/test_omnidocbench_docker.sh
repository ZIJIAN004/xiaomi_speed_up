#!/usr/bin/env bash
# Minimal real Docker + TeX/CDM check; no model inference, dataset scoring or image pull.
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
EVALUATOR_ROOT="${OMNIDOC_EVALUATOR_ROOT:-$ROOT_DIR/evaluation_pipeline/OmniDocBench}"
IMAGE="${OMNIDOC_EVAL_IMAGE:-sunyuefeng/omnidocbench-env:v1.5}"
[[ -f "$EVALUATOR_ROOT/pdf_validation.py" ]] || { echo "Missing evaluator: $EVALUATOR_ROOT" >&2; exit 2; }
EVALUATOR_ROOT=$(cd "$EVALUATOR_ROOT" && pwd)
HEAD_COMMIT=$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)
VERSION_COMMIT=$(git -C "$EVALUATOR_ROOT" rev-parse --verify refs/heads/v1_5 2>/dev/null || git -C "$EVALUATOR_ROOT" rev-parse --verify refs/remotes/origin/v1_5)
[[ "$HEAD_COMMIT" == "$VERSION_COMMIT" ]] || { echo 'Evaluator HEAD is not v1_5.' >&2; exit 2; }
[[ -z "$(git -C "$EVALUATOR_ROOT" status --porcelain --untracked-files=no)" ]] || { echo 'Evaluator tracked files are modified.' >&2; exit 2; }
# Inspect only: never trigger a large implicit download.
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/runs/docker_cdm_smoke_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$RESULT_ROOT" ]] || { echo "Refusing existing output: $RESULT_ROOT" >&2; exit 2; }
mkdir -p "$RESULT_ROOT"
RESULT_ROOT=$(cd "$RESULT_ROOT" && pwd)
printf 'Evaluator commit: %s\nImage: %s\nImage ID: %s\nOutput: %s\n' "$HEAD_COMMIT" "$IMAGE" "$IMAGE_ID" "$RESULT_ROOT" | tee "$RESULT_ROOT/provenance.txt"
docker run --rm --pull=never --network=none \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/opt/omnidocbench \
  --mount "type=bind,src=$EVALUATOR_ROOT,dst=/opt/omnidocbench,readonly" \
  --mount "type=bind,src=$RESULT_ROOT,dst=/opt/omnidocbench/result" \
  --workdir /opt/omnidocbench --entrypoint python "$IMAGE_ID" -c '
import json
from pathlib import Path
import shutil
import subprocess
from metrics.cdm_metric import CDM
for tool in ("xelatex", "magick", "gs", "node"):
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError("Missing required tool: " + tool)
    version = subprocess.run([path,"--version"],check=True,capture_output=True,text=True)
    print(tool+": "+version.stdout.splitlines()[0],flush=True)
# More than three tokens exercises CDM matching/RANSAC, not just a one-symbol render.
formula = r"\frac{x+1}{y+2}=z"
root = Path("result/cdm")
score = CDM(output_root=str(root)).evaluate(formula,formula,"identical_fraction")
print("Actual CDM result: "+json.dumps(score),flush=True)
for side in ("gt","pred"):
    image = root/side/"vis/identical_fraction_base.png"
    bbox = root/side/"bbox/identical_fraction.jsonl"
    if not image.is_file() or image.stat().st_size == 0 or not bbox.is_file() or bbox.stat().st_size == 0:
        raise RuntimeError("Missing actual TeX-rendered image/bbox: "+side)
if score.get("F1_score",0) < .99:
    raise RuntimeError("Identical-formula CDM failed; inspect rendering/runtime, not model accuracy")
Path("result/smoke_result.json").write_text(json.dumps({"status":"passed","formula":formula,"cdm":score},indent=2)+"\n")
print("PASS: Docker, TeX rendering and actual CDM matching",flush=True)
' 2>&1 | tee "$RESULT_ROOT/smoke.log"
