#!/usr/bin/env python3
"""Evaluate IDSpace JSON extraction with the repository's offline vLLM path.

This is generation-based evaluation, not teacher-forced loss evaluation. It
uses benchmark.offline_backend.AsyncLLM, keeps up to --workers requests in
flight, and scores the generated JSON against each manifest target.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from benchmark.offline_backend import OfflineBackend
from benchmark.processor_audit import prepare_image

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

FIELDS = (
    "document_type", "issuing_country", "given_name", "surname", "date_of_birth",
    "gender", "nationality", "personal_identification_number", "document_number",
    "card_num", "personal_num", "birthday", "expire_date", "issue_authority",
    "issue_date", "expiry_date", "place_of_issue",
)


def load_rows(manifest: Path, split: str, max_samples: int) -> list[dict[str, Any]]:
    path = manifest / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    for index, row in enumerate(rows):
        if not isinstance(row.get("image"), str) or not isinstance(row.get("target"), dict):
            raise ValueError(f"{path}:{index + 1} requires image and target fields")
    return rows


def parse_json(text: str) -> tuple[Any | None, str | None]:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as first:
        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value, None
            except json.JSONDecodeError:
                continue
        return None, f"{type(first).__name__}: {first.msg}"


def norm(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def score_record(target: dict[str, Any], text: str) -> dict[str, Any]:
    prediction, error = parse_json(text)
    object_prediction = prediction if isinstance(prediction, dict) else None
    field_results = {}
    for field in FIELDS:
        expected = norm(target.get(field))
        actual = norm(object_prediction.get(field)) if object_prediction is not None else None
        field_results[field] = actual == expected
    present_target_fields = [field for field in FIELDS if field in target]
    valid_field_results = [field_results[field] for field in present_target_fields]
    return {
        "json_valid": object_prediction is not None,
        "json_parse_error": error,
        "prediction": object_prediction,
        "field_results": field_results,
        "field_count": len(present_target_fields),
        "field_correct": sum(valid_field_results),
        "field_exact_match": bool(valid_field_results) and all(valid_field_results),
        "object_exact_match": object_prediction == target,
    }


def make_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(args.model), served_model_name=args.served_model_name,
        max_model_len=args.max_model_len, dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code, workers=args.workers,
        profiler_dir=args.output.parent / (args.output.stem + "_profiler_raw"),
        profile_max_iterations=1, server_processor_dir=args.output.parent / (args.output.stem + "_processor_raw"),
        no_logprobs=True,
    )


def evaluate_split(rows: list[dict[str, Any]], manifest: Path, backend, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    records: list[dict[str, Any]] = []

    def one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        request_started = time.perf_counter()
        image, prep = prepare_image(manifest / row["image"])
        try:
            request_args = SimpleNamespace(
                request_prompt=PROMPT,
                retry_presence_penalty=1.5,
                max_tokens=args.max_tokens,
                no_logprobs=True,
                request_timeout=args.request_timeout,
            )
            result = backend.infer(image, request_args, f"idspace-{index}", 0)
        finally:
            image.close()
        text = result["text"]
        scored = score_record(row["target"], text)
        return {
            "index": index, "image": row["image"], "target": row["target"],
            "raw_text": text, "success": True, "request_seconds": time.perf_counter() - request_started,
            "generated_tokens": result.get("usage", {}).get("completion_tokens"),
            "prompt_tokens": result.get("usage", {}).get("prompt_tokens"),
            "finish_reason": result.get("finish_reason"), "preprocess": prep, **scored,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, index, row) for index, row in enumerate(rows)]
        for future in as_completed(futures):
            records.append(future.result())

    records.sort(key=lambda row: row["index"])
    valid_json = sum(row["json_valid"] for row in records)
    field_total = sum(row["field_count"] for row in records)
    field_correct = sum(row["field_correct"] for row in records)
    return {
        "split": args.split,
        "samples": len(records),
        "successful_requests": sum(row["success"] for row in records),
        "json_valid_rate": valid_json / len(records) if records else 0.0,
        "object_exact_match_rate": sum(row["object_exact_match"] for row in records) / len(records) if records else 0.0,
        "field_exact_match_rate": sum(row["field_exact_match"] for row in records) / len(records) if records else 0.0,
        "field_accuracy": field_correct / field_total if field_total else 0.0,
        "field_correct": field_correct, "field_total": field_total,
        "mean_request_seconds": sum(row["request_seconds"] for row in records) / len(records) if records else None,
        "wall_seconds": time.perf_counter() - started,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--served-model-name", default="Xiaomi-OCR-0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.max_samples < 0 or args.max_tokens < 1:
        parser.error("workers must be positive; max-samples nonnegative; max-tokens positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.gpu.isdecimal():
        parser.error("gpu must be numeric")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    backend_args = make_args(args)
    with OfflineBackend(backend_args) as backend:
        rows = load_rows(args.manifest, args.split, args.max_samples)
        summary = evaluate_split(rows, args.manifest, backend, args)
        payload = {
            "schema": "idspace_vllm_generation_report_v1",
            "model": str(args.model), "manifest": str(args.manifest),
            "split": args.split, "workers": args.workers, "max_tokens": args.max_tokens,
            "prompt": PROMPT, "metric_scope": "generated JSON quality; no teacher-forced loss",
            "backend": backend.metadata, "summary": {k: v for k, v in summary.items() if k != "records"},
            "records": summary["records"],
        }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
