"""Quality adapters. Official OmniDocBench metrics are never replaced by string proxies."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from benchmark.metrics import dump_json, load_jsonl, ratio


def evaluate_baseline(quality):
    """Coverage is not a quality target. Compare only explicitly declared thresholds."""
    thresholds = quality.get("baseline_thresholds", {})
    required = quality.get("required_metrics", [])
    if not isinstance(thresholds, dict):
        thresholds = {}
    comparisons = {}
    for metric in required:
        rule = thresholds.get(metric)
        value = quality.get("metrics", {}).get(metric)
        if not isinstance(rule, dict):
            continue
        target, operator = rule.get("value"), rule.get("operator")
        numeric = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        if not numeric(value) or not numeric(target) or operator not in ("<=", ">="):
            continue
        comparisons[metric] = {"actual": value, "target": target, "operator": operator,
                               "passed": value <= target if operator == "<=" else value >= target}
    covered = bool(required) and len(comparisons) == len(required) and quality.get("status") == "complete"
    return {"status": ("pass" if all(r["passed"] for r in comparisons.values()) else "fail") if covered else "not_evaluated",
            "comparisons": comparisons,
            "reason": None if covered else "complete quality coverage and explicit thresholds for every required metric are needed"}


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def field_quality(manifest, mapping, normalization="none"):
    """Micro exact-match field metrics and reference-normalized CER/WER.

    Fields must be supplied explicitly or extracted by dataset-specific, supplied regexes.
    Ground truth is never used to guess an extraction regex or match a prediction.
    """
    results = {r["sample_id"]: r for r in mapping}
    gt_count = predicted = correct = char_errors = word_errors = chars = words = 0
    evaluated = missing_gt = missing_extractor = 0
    def normalize(value):
        text = str(value)
        return text if normalization == "none" else " ".join(text.casefold().split())
    per_sample = []
    for sample in manifest:
        gt = sample.get("ground_truth_fields")
        if not isinstance(gt, dict) or not gt:
            missing_gt += 1
            continue
        result = results.get(sample["sample_id"], {})
        text = Path(result["prediction_path"]).read_text(encoding="utf-8") if result.get("success") else ""
        fields = None
        # Explicit structured output is preferred. No automatic identity-document schema is guessed.
        if text:
            try:
                value = json.loads(text)
                if isinstance(value, dict) and all(isinstance(v, (str, int, float)) for v in value.values()):
                    fields = value
            except json.JSONDecodeError:
                pass
        patterns = sample.get("field_patterns")
        if fields is None and isinstance(patterns, dict):
            fields = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, text, flags=re.MULTILINE)
                if match:
                    if match.lastindex is None:
                        raise ValueError(f"{sample['sample_id']}: field regex for {key} must contain a capture group")
                    fields[key] = match.group(1)
        if not text:
            fields = {}
        if fields is None:
            missing_extractor += 1
            continue
        gt = {str(k): normalize(v) for k, v in gt.items()}
        fields = {str(k): normalize(v) for k, v in fields.items() if str(v).strip()}
        matched = sum(key in fields and fields[key] == value for key, value in gt.items())
        gt_count += len(gt)
        predicted += len(fields)
        correct += matched
        evaluated += 1
        for key, value in gt.items():
            prediction = fields.get(key, "")
            char_errors += edit_distance(value, prediction)
            word_errors += edit_distance(value.split(), prediction.split())
            chars += len(value)
            words += len(value.split())
        per_sample.append({"sample_id": sample["sample_id"], "ground_truth_fields": len(gt),
                           "predicted_fields": len(fields), "correct_fields": matched})
    precision, recall = ratio(correct, predicted), ratio(correct, gt_count)
    # No predicted fields on an evaluated dataset is a measured zero precision, not missing evidence.
    if evaluated and not predicted:
        precision = 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else (0.0 if evaluated else None)
    return {"status": "complete" if evaluated == len(manifest) and evaluated else "incomplete",
            "source": "explicit_fields_micro_exact_match", "normalization": normalization,
            "evaluated_samples": evaluated, "missing_ground_truth_samples": missing_gt,
            "missing_extractor_samples": missing_extractor,
            "metrics": {"field_accuracy": ratio(correct, gt_count), "precision": precision,
                        "recall": recall, "f1": f1, "cer": ratio(char_errors, chars), "wer": ratio(word_errors, words)},
            "counts": {"correct_fields": correct, "ground_truth_fields": gt_count, "predicted_fields": predicted,
                       "character_edits": char_errors, "reference_characters": chars,
                       "word_edits": word_errors, "reference_words": words},
            "metric_definition": "correct means same key and exact normalized value; CER/WER sum edits over reference fields; extra keys penalize precision",
            "per_sample": per_sample}


def run_quality_command(command_json, run_dir):
    """An argv JSON list, never shell=True. The adapter must write quality.json."""
    output = run_dir / "quality.json"
    command = json.loads(command_json)
    if not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command):
        raise ValueError("quality-command must be a nonempty JSON argv list")
    replacements = {"{manifest}": str(run_dir / "manifest.jsonl"), "{predictions}": str(run_dir / "predictions"),
                    "{mapping}": str(run_dir / "prediction_mapping.json"), "{output}": str(output)}
    for i, arg in enumerate(command):
        for key, value in replacements.items():
            arg = arg.replace(key, value)
        command[i] = arg
    with (run_dir / "quality_evaluator.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode or not output.is_file():
        dump_json(output, {"status": "failed", "returncode": result.returncode,
                           "reason": "quality adapter failed or did not write quality.json; inspect quality_evaluator.log"})
    else:
        data = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(data.get("metrics"), dict):
            raise ValueError("quality adapter must write a metrics object with canonical metric names")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--normalization", choices=("none", "casefold_whitespace"), default="none")
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    dump_json(args.output, field_quality(load_jsonl(args.manifest), mapping, args.normalization))


if __name__ == "__main__":
    main()
