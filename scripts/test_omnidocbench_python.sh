#!/usr/bin/env bash
# Minimal local-Python OmniDocBench v1.5 environment and dataset preflight.
# No model inference, no full scoring, no Docker, no downloads.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVALUATOR_ROOT="${OMNIDOC_EVALUATOR_ROOT:-$ROOT_DIR/evaluation_pipeline/OmniDocBench}"
DATA_ROOT="${OMNIDOC_ROOT:-$ROOT_DIR/benchmark/data}"
EVAL_PYTHON="${OMNIDOC_EVAL_PYTHON:-python}"
TEX_ROOT="${TEXLIVE_ROOT:-}"
EVAL_BIN=$(dirname "$(command -v "$EVAL_PYTHON" 2>/dev/null || printf '%s' "$EVAL_PYTHON")")
export PATH="$EVAL_BIN:$PATH"
export PYTHONPATH="$ROOT_DIR:$EVALUATOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 2; }
ok() { printf 'OK: %s\n' "$*"; }

[[ -f "$EVALUATOR_ROOT/pdf_validation.py" ]] || fail "missing evaluator: $EVALUATOR_ROOT/pdf_validation.py"
[[ -f "$EVALUATOR_ROOT/requirements.txt" ]] || fail "missing evaluator requirements.txt"
command -v git >/dev/null 2>&1 || fail "git is unavailable"
command -v "$EVAL_PYTHON" >/dev/null 2>&1 || fail "Python is unavailable: $EVAL_PYTHON"

HEAD_COMMIT=$(git -C "$EVALUATOR_ROOT" rev-parse HEAD)
VERSION_COMMIT=$(git -C "$EVALUATOR_ROOT" rev-parse --verify refs/heads/v1_5 2>/dev/null || git -C "$EVALUATOR_ROOT" rev-parse --verify refs/remotes/origin/v1_5)
[[ "$HEAD_COMMIT" == "$VERSION_COMMIT" ]] || fail "evaluator HEAD is not the pinned v1_5 commit"
[[ -z "$(git -C "$EVALUATOR_ROOT" status --porcelain --untracked-files=no)" ]] || fail "evaluator tracked files are modified"
ok "clean OmniDocBench v1_5 evaluator $HEAD_COMMIT"

"$EVAL_PYTHON" - "$EVALUATOR_ROOT" <<'PY'
import importlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for module in ("dataset", "task", "metrics", "metrics.cdm_metric", "skimage.measure"):
    importlib.import_module(module)
print("OK: evaluator Python imports, including skimage.measure.ransac")
PY

for tool in xelatex magick gs node; do
    command -v "$tool" >/dev/null 2>&1 || fail "missing external tool: $tool"
    ok "$tool $("$tool" --version 2>/dev/null | head -n 1)"
done

for file in upgreek.sty xeCJK.sty geometry.sty amsmath.sty amssymb.sty xcolor.sty; do
    kpsewhich "$file" >/dev/null 2>&1 || fail "missing TeX file: $file"
    ok "TeX file $file"
done

if command -v fc-match >/dev/null 2>&1; then
    FONT_MATCH=$(fc-match --format='%{family}\n' "Source Han Sans SC" 2>/dev/null || true)
    printf '%s\n' "$FONT_MATCH" | grep -Fqi "Source Han Sans SC" \
        || fail "Source Han Sans SC is unavailable (fontconfig fallback: ${FONT_MATCH:-none})"
    ok "Source Han Sans SC: $FONT_MATCH"
else
    fail "fc-match is unavailable; cannot verify CDM CJK font"
fi

[[ -d "$DATA_ROOT/images" ]] || fail "missing dataset images directory: $DATA_ROOT/images"
[[ -f "$DATA_ROOT/OmniDocBench.json" ]] || fail "missing ground truth: $DATA_ROOT/OmniDocBench.json"

"$EVAL_PYTHON" - "$DATA_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
with (root / "OmniDocBench.json").open(encoding="utf-8") as handle:
    gt = json.load(handle)
if not isinstance(gt, list):
    raise SystemExit("FAIL: OmniDocBench.json is not a list")
paths = []
for row in gt:
    page_info = row.get("page_info") if isinstance(row, dict) else None
    image_path = page_info.get("image_path") if isinstance(page_info, dict) else None
    if not isinstance(image_path, str):
        raise SystemExit("FAIL: GT row lacks page_info.image_path")
    paths.append(pathlib.Path(image_path.replace("\\", "/")).name)
files = {p.name for p in (root / "images").iterdir() if p.is_file()}
missing = sorted(set(paths) - files)
duplicates = len(paths) - len(set(paths))
if missing:
    raise SystemExit(f"FAIL: {len(missing)} GT images are missing; first={missing[:3]}")
if duplicates:
    raise SystemExit(f"FAIL: GT contains {duplicates} duplicate image_path entries")
print(f"OK: dataset records={len(gt)} images={len(files)} GT-image coverage={len(paths)}/{len(gt)}")
PY

"$EVAL_PYTHON" - "$EVALUATOR_ROOT" <<'PY'
import json
import pathlib
import sys
import tempfile
import benchmark.skimage_compat
from metrics.cdm_metric import CDM

root = pathlib.Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="omnidocbench_cdm_preflight_") as directory:
    out = pathlib.Path(directory)
    formula = r"\frac{x+1}{y+2}=z"
    score = CDM(output_root=str(out)).evaluate(formula, formula, "identical_formula")
    if score.get("F1_score", 0) < 0.99:
        raise SystemExit(f"FAIL: identical-formula CDM score={score}")
    for side in ("gt", "pred"):
        image = out / side / "vis" / "identical_formula_base.png"
        bbox = out / side / "bbox" / "identical_formula.jsonl"
        if not image.is_file() or image.stat().st_size == 0:
            raise SystemExit(f"FAIL: missing rendered image: {image}")
        if not bbox.is_file() or bbox.stat().st_size == 0:
            raise SystemExit(f"FAIL: missing bbox output: {bbox}")
    print("OK: real CDM identical-formula render/match", json.dumps(score))
PY

printf 'PASS: Python evaluator, TeX/CDM runtime, and dataset preflight\n'
printf 'Evaluator commit: %s\n' "$HEAD_COMMIT"
printf 'Dataset root: %s\n' "$DATA_ROOT"
