#!/usr/bin/env python3
"""Score generated IDSpace JSON with overall, field, and subgroup metrics."""
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
GROUPS = ("country", "modality", "template_id")
# The 12 fields the IDSpace prompt asks for. Labels also carry birthday/card_num/expire_date/issue_authority/
# personal_num, always null and never requested; scoring them would count "absent == null" as correct.
ASKED_FIELDS = (
    "document_type", "issuing_country", "given_name", "surname", "date_of_birth", "gender", "nationality",
    "personal_identification_number", "document_number", "issue_date", "expiry_date", "place_of_issue",
)
DATE_FIELDS = {"date_of_birth", "issue_date", "expiry_date"}
# Not scored for now: labels use a fixed vocabulary the prompt never states (document_type is always
# "identity_card"; issuing_country is an ISO3 code), so the model copies card text (ESPAÑA, Identity Card)
# and exact match measures format, not reading. The country breakdown below uses the manifest's
# "country" metadata, not this field, so per-country statistics are unaffected.
UNSCORED_FIELDS = {"document_type", "issuing_country"}
SCORED_FIELDS = tuple(field for field in ASKED_FIELDS if field not in UNSCORED_FIELDS)


def parse_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            return value[0], None  # model wrapped the single object in a list: [{...}]
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


def norm_date(value: Any) -> Any:
    """DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY / 'DD MM YYYY' -> YYYY-MM-DD (day-first, as on the IDSpace cards)."""
    value = norm(value)
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"(\d{1,2})[.\-/ ](\d{1,2})[.\-/ ](\d{4})", value)
    return f"{match[3]}-{int(match[2]):02d}-{int(match[1]):02d}" if match else value


def new_bucket() -> dict[str, Any]:
    return {
        "samples": 0, "json_valid": 0, "object_exact_match": 0,
        "field_exact_match": 0, "correct_fields": 0, "ground_truth_fields": 0,
        "per_field": defaultdict(lambda: {"correct": 0, "total": 0}),
    }


def add_record(bucket: dict[str, Any], record: dict[str, Any], fields: list[str]) -> None:
    bucket["samples"] += 1
    bucket["json_valid"] += int(record["json_valid"])
    bucket["object_exact_match"] += int(record["object_exact_match"])
    bucket["field_exact_match"] += int(record["field_exact_match"])
    bucket["correct_fields"] += record["field_correct"]
    bucket["ground_truth_fields"] += record["field_total"]
    for field in fields:
        stats = bucket["per_field"][field]
        stats["total"] += 1
        stats["correct"] += int(record["field_results"][field])


def finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    samples = bucket["samples"]
    total_fields = bucket["ground_truth_fields"]
    result = {
        "samples": samples,
        "json_valid_rate": bucket["json_valid"] / samples if samples else None,
        "object_exact_match_rate": bucket["object_exact_match"] / samples if samples else None,
        "field_exact_match_rate": bucket["field_exact_match"] / samples if samples else None,
        "field_accuracy": bucket["correct_fields"] / total_fields if total_fields else None,
        "counts": {key: bucket[key] for key in (
            "json_valid", "object_exact_match", "field_exact_match",
            "correct_fields", "ground_truth_fields")},
        "per_field": {
            field: {"correct": values["correct"], "total": values["total"],
                    "accuracy": values["correct"] / values["total"] if values["total"] else None}
            for field, values in sorted(bucket["per_field"].items())
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    overall = new_bucket()
    grouped = {group: defaultdict(new_bucket) for group in GROUPS}
    per_sample = []

    for row in rows:
        sample_id = row["sample_id"]
        target = row.get("ground_truth_fields")
        if not isinstance(target, dict):
            raise ValueError(f"{args.manifest}: sample {sample_id} lacks ground_truth_fields")
        prediction_path = args.predictions / f"{sample_id}.md"
        text = prediction_path.read_text(encoding="utf-8") if prediction_path.is_file() else ""
        prediction, error = parse_json(text)
        # Score only requested fields that have a real label; a null label is not a scoreable target.
        fields = [field for field in SCORED_FIELDS if norm(target.get(field)) not in (None, "")]
        results = {}
        correct = 0
        for field in fields:
            convert = norm_date if field in DATE_FIELDS else norm
            expected = convert(target.get(field))
            actual = convert(prediction.get(field)) if prediction is not None else None
            results[field] = actual == expected
            correct += int(results[field])
        record = {
            "sample_id": sample_id,
            "image": row["images"][0],
            "country": row.get("country", "unknown"),
            "modality": row.get("modality", "unknown"),
            "template_id": row.get("template_id", "unknown"),
            "json_valid": prediction is not None,
            "json_parse_error": error,
            "prediction": prediction,
            "target": target,
            "object_exact_match": prediction == target if prediction is not None else False,
            "field_exact_match": bool(fields) and correct == len(fields),
            "field_correct": correct,
            "field_total": len(fields),
            "field_results": results,
        }
        add_record(overall, record, fields)
        for group in GROUPS:
            add_record(grouped[group][record[group]], record, fields)
        per_sample.append(record)

    output = {
        "status": "complete" if rows else "incomplete",
        "source": "idspace_generated_json_exact_match",
        "input_samples": len(rows),
        "evaluated_samples": len(rows),
        "metrics": finalize_bucket(overall),
        "by_group": {
            group: {key: finalize_bucket(bucket) for key, bucket in sorted(values.items(), key=lambda item: str(item[0]))}
            for group, values in grouped.items()
        },
        "per_sample": per_sample,
        "unscored_fields": sorted(UNSCORED_FIELDS),
        "metric_definition": "JSON parsing strips an optional markdown fence, unwraps a single-object list [{...}], and accepts the first JSON object; only the prompt-requested fields with a non-null label are scored (null labels and never-requested label keys are excluded); document_type and issuing_country are not scored because labels use a vocabulary the prompt does not state; by_group.country comes from manifest metadata, independent of issuing_country; dates are normalized day-first to YYYY-MM-DD on both sides; other values use exact equality after string trim; missing fields are incorrect; extra prediction fields do not affect field_accuracy but prevent object_exact_match. Each subgroup reports sample-level JSON validity/exact match, micro field accuracy, and per-field accuracy.",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
