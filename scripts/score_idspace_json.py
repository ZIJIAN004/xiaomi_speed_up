#!/usr/bin/env python3
"""Score generated IDSpace JSON predictions for benchmark.run_benchmark."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

FIELDS = (
    "document_type", "issuing_country", "given_name", "surname", "date_of_birth",
    "gender", "nationality", "personal_identification_number", "document_number",
    "card_num", "personal_num", "birthday", "expire_date", "issue_authority",
    "issue_date", "expiry_date", "place_of_issue",
)


def parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        return (value, None) if isinstance(value, dict) else (None, "top-level JSON is not an object")
    except json.JSONDecodeError as first:
        decoder = json.JSONDecoder()
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return (value, None) if isinstance(value, dict) else (None, "first JSON value is not an object")
            except json.JSONDecodeError:
                continue
        return None, f"JSONDecodeError: {first.msg}"


def norm(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_sample = []
    field_stats = defaultdict(lambda: [0, 0])
    valid_json = object_exact = sample_field_exact = field_correct = field_total = 0
    for row in rows:
        sample_id = row["sample_id"]
        target = row.get("ground_truth_fields")
        if not isinstance(target, dict):
            raise ValueError(f"{args.manifest}: sample {sample_id} lacks ground_truth_fields")
        prediction_path = args.predictions / f"{sample_id}.md"
        text = prediction_path.read_text(encoding="utf-8") if prediction_path.is_file() else ""
        prediction, error = parse_json(text)
        if prediction is not None:
            valid_json += 1
        object_match = prediction == target if prediction is not None else False
        object_exact += object_match
        fields = [field for field in FIELDS if field in target]
        results = {}
        correct = 0
        for field in fields:
            expected, actual = norm(target.get(field)), norm(prediction.get(field)) if prediction is not None else None
            matched = actual == expected
            results[field] = matched
            field_stats[field][1] += 1
            if matched:
                correct += 1
                field_stats[field][0] += 1
        field_correct += correct
        field_total += len(fields)
        exact = bool(fields) and correct == len(fields)
        sample_field_exact += exact
        per_sample.append({
            "sample_id": sample_id,
            "image": row["images"][0],
            "json_valid": prediction is not None,
            "json_parse_error": error,
            "prediction": prediction,
            "target": target,
            "object_exact_match": object_match,
            "field_exact_match": exact,
            "field_correct": correct,
            "field_total": len(fields),
            "field_results": results,
        })

    count = len(rows)
    metrics = {
        "json_valid_rate": valid_json / count if count else None,
        "object_exact_match_rate": object_exact / count if count else None,
        "field_exact_match_rate": sample_field_exact / count if count else None,
        "field_accuracy": field_correct / field_total if field_total else None,
    }
    output = {
        "status": "complete" if count else "incomplete",
        "source": "idspace_generated_json_exact_match",
        "input_samples": count,
        "evaluated_samples": count,
        "metrics": metrics,
        "counts": {"json_valid": valid_json, "object_exact_match": object_exact,
                   "field_exact_match": sample_field_exact, "correct_fields": field_correct,
                   "ground_truth_fields": field_total},
        "per_field": {
            field: {"correct": values[0], "total": values[1],
                    "accuracy": values[0] / values[1] if values[1] else None}
            for field, values in sorted(field_stats.items())
        },
        "per_sample": per_sample,
        "metric_definition": "JSON parsing strips an optional markdown fence and accepts the first JSON object; field values use exact equality after string trim; missing fields are incorrect; extra prediction fields do not affect field_accuracy but prevent object_exact_match.",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
