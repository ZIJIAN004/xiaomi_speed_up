#!/usr/bin/env python3
"""Run the IDSpace task through benchmark.run_benchmark's full vLLM instrumentation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from benchmark.metrics import dump_json
from benchmark.omnidocbench_v15 import write_jsonl
from benchmark.run_benchmark import main as run_benchmark, parse_args as parse_benchmark_args

PROMPT = """Extract identity information from this ID card.

Instructions:
1. Extract only information visible in the image.
2. Do not infer missing information.
3. Preserve original names and identification numbers.
4. Normalize unambiguous dates to YYYY-MM-DD.
5. Return null for missing or unreadable fields.
6. Return only valid JSON.

Required fields:
{
  \"document_type\": null,
  \"issuing_country\": null,
  \"given_name\": null,
  \"surname\": null,
  \"date_of_birth\": null,
  \"gender\": null,
  \"nationality\": null,
  \"personal_identification_number\": null,
  \"document_number\": null,
  \"issue_date\": null,
  \"expiry_date\": null,
  \"place_of_issue\": null
}"""


def load_rows(manifest: Path, split: str, max_samples: int):
    path = manifest / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    return rows


def make_benchmark_manifest(rows, source_root: Path, output: Path):
    samples = []
    for index, row in enumerate(rows):
        image = (source_root / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        sample_id = f"idspace_{index:08d}"
        samples.append({
            "sample_id": sample_id,
            "dataset": "idspace",
            "benchmark_version": "idspace",
            # `region` selects the runner's text-preserving output path; this is
            # an internal storage mode, not a change to the IDSpace task.
            "request_unit": "region",
            "images": [str(image)],
            "relative_path": row["image"],
            "prompt": PROMPT,
            "ground_truth_fields": row["target"],
        })
    write_jsonl(output, samples)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="IDSpace prepared directory")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New benchmark result directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--representatives", type=int, default=5)
    parser.add_argument("--profile-max-iterations", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-processor-audit", action="store_true")
    parser.add_argument("--output-retention", choices=("sampled", "all"), default="all")
    args = parser.parse_args()
    if args.workers < 1 or args.max_samples < 0 or args.max_tokens < 1:
        parser.error("workers must be positive; max-samples nonnegative; max-tokens positive")
    if args.representatives < 3:
        parser.error("representatives must be at least 3")
    if args.output.exists():
        parser.error("output must be a new directory")
    if not args.gpu.isdecimal():
        parser.error("gpu must be numeric")

    rows = load_rows(args.manifest, args.split, args.max_samples)
    if not rows:
        parser.error("selected split is empty")
    args.output.mkdir(parents=True)
    benchmark_manifest = args.output / "idspace_manifest.jsonl"
    samples = make_benchmark_manifest(rows, args.manifest, benchmark_manifest)
    targets = {sample["sample_id"]: row["target"] for sample, row in zip(samples, rows)}
    dump_json(args.output / "idspace_targets.json", targets)

    argv = [
        "--manifest", str(benchmark_manifest), "--output", str(args.output / "inference"),
        "--model", str(args.model), "--served-model-name", "Xiaomi-OCR-0",
        "--backend", "offline", "--gpu", args.gpu, "--workers", str(args.workers),
        "--warmup", str(args.warmup), "--representatives", str(min(args.representatives, len(samples))),
        "--max-tokens", str(args.max_tokens), "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--profile-max-iterations", str(args.profile_max_iterations),
        "--output-retention", args.output_retention,
        "--quality-command", json.dumps([
            os.environ.get("IDSPACE_SCORER_PYTHON", os.sys.executable),
            str(Path(__file__).with_name("score_idspace_json.py")),
            "--manifest", "{manifest}", "--predictions", "{predictions}", "--output", "{output}",
        ]),
    ]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.skip_processor_audit:
        argv.append("--skip-processor-audit")
    if args.output_retention == "all":
        argv.extend(["--retain-samples", "0", "--retain-worst", "0"])
    else:
        argv.extend(["--retain-samples", "10", "--retain-worst", "10"])

    benchmark_args = parse_benchmark_args(argv)
    run_benchmark(benchmark_args)
    inference = args.output / "inference"
    quality = inference / "quality.json"
    if quality.is_file():
        shutil.copy2(quality, args.output / "quality.json")
    summary = inference / "summary.json"
    if summary.is_file():
        shutil.copy2(summary, args.output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
