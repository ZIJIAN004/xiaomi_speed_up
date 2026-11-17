#!/usr/bin/env python3
"""Collect lightweight request/output metrics from the three benchmark runs."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

DATASETS = ("omnidocbench", "midv500", "midv-lait")
TOKEN_RE = re.compile(r"\S+")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def summarize(dataset: str, prediction_dir: Path, manifest: Path, elapsed_s: float) -> dict:
    expected = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    outputs = sorted(prediction_dir.glob("*.md")) if prediction_dir.exists() else []
    output_tokens = [len(TOKEN_RE.findall(p.read_text(encoding="utf-8", errors="replace"))) for p in outputs]
    elapsed = max(elapsed_s, 0.0)
    return {
        "dataset": dataset,
        "expected_images": expected,
        "output_files": len(outputs),
        "missing_outputs": max(expected - len(outputs), 0),
        "error_rate": (max(expected - len(outputs), 0) / expected) if expected else None,
        "elapsed_s": elapsed,
        "images_per_second": (len(outputs) / elapsed) if elapsed else None,
        "output_token_mean_word_proxy": statistics.mean(output_tokens) if output_tokens else None,
        "output_token_p50_word_proxy": percentile(output_tokens, 0.50),
        "output_token_p95_word_proxy": percentile(output_tokens, 0.95),
        "output_token_p99_word_proxy": percentile(output_tokens, 0.99),
        "note": "Output token fields here are only a fallback sanity proxy; exact token metrics require vLLM usage/token logs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    args = parser.parse_args()
    records = []
    for name in DATASETS:
        elapsed_file = args.result_root / f"{name}.elapsed_s"
        elapsed = float(elapsed_file.read_text().strip()) if elapsed_file.exists() else 0.0
        records.append(summarize(name, args.result_root / "predictions" / name,
                                 args.result_root / "manifests" / f"{name}.jsonl", elapsed))
    summary = {
        "model": args.model,
        "gpu": args.gpu,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "datasets": records,
        "warning": "This collector does not calculate OmniDocBench/MIDV quality scores. Run the official evaluators with their ground truth files.",
    }
    path = args.result_root / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
