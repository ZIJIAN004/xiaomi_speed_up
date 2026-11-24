"""Shared metric semantics. Missing observations stay null, never synthetic zero."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

STAGES = (
    "file_io", "image_decode", "preprocess", "processor", "h2d",
    "vision_encoder", "projector", "prefill", "decode", "sampling", "draft",
    "tokenizer_decode", "postprocess", "serialization", "output_io",
)
TOKEN_FIELDS = (
    "system_tokens", "user_text_tokens", "template_tokens", "special_input_tokens",
    "prompt_text_tokens", "input_ids_length", "image_patch_count", "visual_tokens",
    "total_input_tokens", "generated_tokens", "generated_visible_tokens",
    "generated_special_tokens", "total_processed_tokens",
)
STAGE_ALIASES = {"runner_prefill": "prefill", "runner_decode": "decode"}

PADDING_FIELDS = (
    "prompt_length_i", "max_prompt_length", "prompt_padding_tokens", "prompt_padding_ratio",
    "visual_length_i", "max_visual_tokens_in_batch", "visual_padding_tokens",
    "visual_padding_ratio", "image_slot_padding", "pixel_padding_area", "pixel_padding_ratio",
    "estimated_visual_padding_tokens", "generated_length_i", "max_generated_length",
    "decode_padding_tokens", "decode_padding_ratio",
)


def percentile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v)]
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.pstdev(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            **{f"p{p}": percentile(values, p / 100) for p in (50, 90, 95, 99)}}


def cuda_event_statistics(worker_results):
    """Aggregate exclusive CUDA-event and host-wall spans without synthetic zeros."""
    totals = {}
    worker_ttft = []
    host_totals = {}
    for worker in worker_results:
        events = [e for e in worker.get("events", []) if isinstance(e.get("start_ms"), (int, float))
                  and isinstance(e.get("end_ms"), (int, float)) and e["end_ms"] >= e["start_ms"]]
        children = {}
        for e in events:
            children.setdefault((e.get("parent_id"), e["stream"]), []).append((e["start_ms"], e["end_ms"]))
        for e in events:
            inclusive = e["end_ms"] - e["start_ms"]
            nested = [(max(a, e["start_ms"]), min(b, e["end_ms"]))
                      for a, b in children.get((e["id"], e["stream"]), [])]
            exclusive = max(0, inclusive - union_duration(nested))
            stage = totals.setdefault(e["stage"], {"count": 0, "inclusive_ms": 0.0, "exclusive_ms": 0.0})
            stage["count"] += 1
            stage["inclusive_ms"] += inclusive
            stage["exclusive_ms"] += exclusive
            host_span = e.get("host_call_span_ms")
            if isinstance(host_span, (int, float)):
                host_stage = host_totals.setdefault(e["stage"], {"count": 0, "inclusive_ms": 0.0})
                host_stage["count"] += 1
                host_stage["inclusive_ms"] += host_span
        prefill = [e for e in events if e["stage"] == "prefill"]
        first_token = [e for e in events if e.get("final_prefill_sampling")]
        if prefill and first_token and len({e["stream"] for e in prefill}) == 1:
            first = min(prefill, key=lambda e: e["start_ms"])
            last = min((e for e in first_token if e["stream"] == first["stream"] and e["end_ms"] >= first["start_ms"]),
                       key=lambda e: e["end_ms"], default=None)
            if last:
                worker_ttft.append(last["end_ms"] - first["start_ms"])
    stage_sources = {}
    host_call_span_sources = {}
    for stage in ("prefill", "decode"):
        runner = "runner_" + stage
        if stage not in totals and runner in totals:
            totals[stage] = dict(totals[runner])
            stage_sources[stage] = "runner_phase_cuda_event_fallback"
        elif stage in totals:
            stage_sources[stage] = "nested_stage_cuda_event"
        if stage not in host_totals and runner in host_totals:
            host_totals[stage] = dict(host_totals[runner])
            host_call_span_sources[stage] = "runner_phase_host_call_span_fallback"
    for stage in host_totals:
        host_call_span_sources.setdefault(stage, "representative_worker_host_call_span_inclusive")
    return {"stages": totals, "host_call_spans": host_totals, "stage_sources": stage_sources,
            "host_call_span_sources": host_call_span_sources,
            "gpu_ttft_ms": max(worker_ttft) if worker_ttft else None,
            "gpu_ttft_source": "first model prefill event to final-prefill sampling completion on same CUDA stream",
            "scope": "representative with profiler; per-stream elapsed spans, not pure kernel work; nested stages exclusive",
            "host_call_span_scope": "representative worker call spans; includes CPU/runtime waits and is not additive under overlap"}


def weighted_stats(counts):
    counts = {float(k): int(v) for k, v in counts.items() if int(v) > 0}
    n = sum(counts.values())
    if not n:
        return stats([])
    ordered = sorted(counts)
    mean = sum(k * v for k, v in counts.items()) / n
    def at(index):
        total = 0
        for key in ordered:
            total += counts[key]
            if index < total:
                return key
    def quantile(q):
        index = (n - 1) * q
        lo = int(index)
        return at(lo) + (at(min(lo + 1, n - 1)) - at(lo)) * (index - lo)
    return {"count": n, "mean": mean, "std": math.sqrt(sum(v * (k - mean) ** 2 for k, v in counts.items()) / n),
            "min": ordered[0], "max": ordered[-1], **{f"p{p}": quantile(p / 100) for p in (50, 90, 95, 99)}}


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def exact_sum(records, field):
    values = [r.get(field) for r in records]
    return sum(values) if values and all(v is not None for v in values) else None


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def area_bucket(mp):
    if mp is None:
        return "unknown"
    for bound in (1, 2, 4, 8):
        if mp <= bound:
            return {1: "0-1 MP", 2: "1-2 MP", 4: "2-4 MP", 8: "4-8 MP"}[bound]
    return ">8 MP"


def output_bucket(n):
    if n is None:
        return "unknown"
    for bound in (128, 256, 512, 1024):
        if n <= bound:
            return {128: "0-128", 256: "129-256", 512: "257-512", 1024: "513-1024"}[bound]
    return ">1024"


def length_bucket(n):
    if n is None:
        return "unknown"
    for bound in (256, 512, 1024, 2048, 4096, 8192):
        if n <= bound:
            return f"<={bound}"
    return ">8192"


def union_duration(intervals):
    intervals = sorted((a, b) for a, b in intervals if b > a)
    total = 0.0
    end = None
    for a, b in intervals:
        total += b - max(a, end if end is not None else a) if end is None or b > end else 0
        end = max(end, b) if end is not None else b
    return total


def overlap_duration(left, right):
    # Sweep the boundaries of each interval union, avoiding double counting nested CPU ops.
    def merged(items):
        result = []
        for a, b in sorted(items):
            if b <= a:
                continue
            if result and a <= result[-1][1]:
                result[-1] = (result[-1][0], max(result[-1][1], b))
            else:
                result.append((a, b))
        return result
    left, right = merged(left), merged(right)
    i = j = 0
    result = 0
    while i < len(left) and j < len(right):
        result += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result
