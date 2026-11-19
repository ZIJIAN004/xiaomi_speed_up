#!/usr/bin/env python3
"""Re-run OmniDocBench v1.5 on an existing prediction directory and diagnose missing page results.

This script never runs model inference and never modifies the original result directory.
It creates an isolated evaluator_diagnosis_* directory below the existing result root,
runs the official evaluator with the same metric configuration, preserves stdout/stderr,
and reports which pages are absent from each official per-page output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

METRICS = {
    "text_block": {"metric": ["Edit_dist"]},
    "display_formula": {"metric": ["Edit_dist", "CDM"]},
    "table": {"metric": ["TEDS", "Edit_dist"]},
    "reading_order": {"metric": ["Edit_dist"]},
}
RESULT_FILES = {
    "text_block": "predictions_quick_match_text_block_per_page_edit.json",
    "display_formula": "predictions_quick_match_display_formula_per_page_edit.json",
    "table": "predictions_quick_match_table_per_page_edit.json",
    "reading_order": "predictions_quick_match_reading_order_per_page_edit.json",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stem(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem


def gt_stems(gt_path: Path) -> set[str]:
    rows = read_json(gt_path)
    return {
        stem(row["page_info"]["image_path"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("page_info"), dict)
        and isinstance(row["page_info"].get("image_path"), str)
    }


def official_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = read_json(path)
    # v1_5 per-page Edit_dist outputs are dictionaries keyed by image name/path.
    # For a schema variant, inspect only explicit image identity fields in rows;
    # never scan arbitrary metric/text strings, which would create false hits.
    if isinstance(payload, dict):
        return {stem(key) for key in payload if isinstance(key, str)}
    if isinstance(payload, list):
        keys = set()
        identity_fields = ("image_name", "image_path", "img_id", "img_name", "image")
        for row in payload:
            if isinstance(row, dict):
                for field in identity_fields:
                    value = row.get(field)
                    if isinstance(value, str):
                        keys.add(stem(value))
        return keys
    return set()


def safe_link(source: Path, target: Path):
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def build_config(dataset: Path, predictions: Path, config_path: Path):
    config = {
        "end2end_eval": {
            "metrics": METRICS,
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(dataset / "OmniDocBench.json")},
                "prediction": {"data_path": str(predictions)},
                "match_method": "quick_match",
            },
        }
    }
    dump_json(config_path, config)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path,
                        help="completed run directory containing inference/predictions")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evaluator-root", required=True, type=Path)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--texlive-root", default="")
    args = parser.parse_args(argv)

    result = args.result.resolve()
    dataset = args.dataset.resolve()
    evaluator = args.evaluator_root.resolve()
    predictions = result / "inference" / "predictions"
    gt = dataset / "OmniDocBench.json"
    if not result.is_dir():
        parser.error(f"missing result directory: {result}")
    if not predictions.is_dir() or not gt.is_file():
        parser.error("result/inference/predictions or dataset/OmniDocBench.json is missing")
    if not (evaluator / "pdf_validation.py").is_file():
        parser.error(f"missing official evaluator: {evaluator / 'pdf_validation.py'}")

    quality_path = result / "quality.json"
    quality = read_json(quality_path) if quality_path.is_file() else {}
    expected = gt_stems(gt)
    predicted = {path.stem for path in predictions.glob("*.md")}
    missing = set(quality.get("unobserved_evaluator_pages", []))
    if not missing:
        missing = expected - predicted

    diagnosis = result / f"evaluator_diagnosis_{time.strftime('%Y%m%d_%H%M%S')}"
    work = diagnosis / "work"
    eval_result = work / "result"
    diagnosis.mkdir()
    work.mkdir()
    eval_result.mkdir()

    # The official script resolves resources relative to its working directory.
    for asset in evaluator.iterdir():
        if asset.name not in {".git", "result", "__pycache__"}:
            safe_link(asset, work / asset.name)

    config_path = diagnosis / "config.json"
    build_config(dataset, predictions, config_path)
    invocation = diagnosis / "invocation.json"

    env = dict(os.environ)
    project_root = Path(__file__).resolve().parent.parent
    eval_python = shutil.which(args.eval_python) or args.eval_python
    eval_bin = str(Path(eval_python).resolve().parent)
    tex_root = args.texlive_root or env.get("TEXLIVE_ROOT", "/home/mr_jiang/tools/texlive")
    tex_bin = str(Path(tex_root) / "bin" / "x86_64-linux")
    env["PYTHONPATH"] = os.pathsep.join((str(project_root), str(evaluator), env.get("PYTHONPATH", "")))
    env["PATH"] = os.pathsep.join((eval_bin, tex_bin, env.get("PATH", "")))
    env["TEXLIVE_ROOT"] = tex_root
    env["TEXMFROOT"] = tex_root
    env["TEXMFCNF"] = str(Path(tex_root) / "texmf-dist" / "web2c")

    script = evaluator / "pdf_validation.py"
    command = [
        eval_python, "-c",
        "import runpy, sys; import benchmark.skimage_compat; "
        "script, config = sys.argv[1:3]; "
        "sys.argv = [script, '--config', config]; "
        "runpy.run_path(script, run_name='__main__')",
        str(script), str(config_path),
    ]
    dump_json(invocation, {"command": command, "cwd": str(work), "env_paths": {
        "python": eval_python, "texlive": tex_root, "evaluator": str(evaluator)
    }})

    log_path = diagnosis / "evaluator.log"
    print(f"DIAGNOSIS_DIR={diagnosis}")
    print(f"RUNNING_OFFICIAL_EVALUATOR={evaluator / 'pdf_validation.py'}")
    print(f"PREDICTION_COUNT={len(predicted)} EXPECTED_COUNT={len(expected)}")
    print(f"TARGET_MISSING_COUNT={len(missing)}")
    completed = subprocess.run(command, cwd=work, env=env, stdout=log_path.open("w", encoding="utf-8"),
                               stderr=subprocess.STDOUT, check=False)
    print(f"EVALUATOR_RETURN_CODE={completed.returncode}")
    print(f"EVALUATOR_LOG={log_path}")

    report = {
        "result": str(result),
        "diagnosis": str(diagnosis),
        "return_code": completed.returncode,
        "expected_count": len(expected),
        "prediction_count": len(predicted),
        "target_missing_pages": sorted(missing),
        "official_outputs": {},
        "target_page_presence": {},
        "log_hits": [],
    }
    for category, filename in RESULT_FILES.items():
        path = eval_result / filename
        keys = official_keys(path)
        report["official_outputs"][category] = {
            "path": str(path), "exists": path.is_file(), "key_count": len(keys)
        }
        report["target_page_presence"][category] = sorted(missing & keys)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = re.compile(r"Traceback|Exception|Error|ERROR|failed|skip|missing|CDM|TEDS|latex|compile", re.I)
    report["log_hits"] = [
        {"line": index, "text": line}
        for index, line in enumerate(log_text.splitlines(), 1)
        if patterns.search(line)
    ][-300:]
    dump_json(diagnosis / "diagnosis_report.json", report)

    print("\nOFFICIAL_OUTPUT_COVERAGE")
    for category, info in report["official_outputs"].items():
        print(f"{category}: exists={info['exists']} key_count={info['key_count']} "
              f"target_present={len(report['target_page_presence'][category])}")
    print(f"LOG_SUSPICIOUS_LINES={len(report['log_hits'])}")
    for hit in report["log_hits"][-80:]:
        print(f"{hit['line']}: {hit['text']}")
    print(f"REPORT={diagnosis / 'diagnosis_report.json'}")
    print("TARGET_PAGE_DETAILS")
    for page in sorted(missing):
        present = [category for category in RESULT_FILES
                   if page in report["target_page_presence"][category]]
        print(f"{page}\t{','.join(present) if present else 'ABSENT_FROM_ALL_PER_PAGE_OUTPUTS'}")

    # Nonzero is reserved for evaluator execution failure. Missing-page coverage is
    # reported in the diagnosis, because that is precisely what this script diagnoses.
    return 0 if completed.returncode == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
