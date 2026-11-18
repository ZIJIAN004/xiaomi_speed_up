"""Aggregate actual processor calls by phase windows, never request completion order."""
from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import dump_json, stats, write_csv


def collect_processor_calls(run_dir, config, records, warmups=(), profiles=()):
    run_dir = Path(run_dir)
    directory = config.get("server_processor_dir")
    rows, metadata, errors = [], [], []
    windows = [("steady", config["steady_start_s"], config["steady_end_s"])]
    populations = [("warmup", warmups), ("profile", profiles)]
    retries_path = run_dir / "retry_metrics.json"
    if retries_path.is_file():
        populations.append(("retry", json.loads(retries_path.read_text(encoding="utf-8"))))
    for phase, population in populations:
        windows += [(phase, r["request_start_s"], r["request_end_s"]) for r in population
                    if r.get("request_start_s") is not None and r.get("request_end_s") is not None]
    if directory:
        for path in sorted(Path(directory).glob("processor_*.jsonl")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    row = json.loads(line)
                    if row.get("event") != "call":
                        metadata.append(row)
                        continue
                    start, end = row["start_s"], row["end_s"]
                    matches = [phase for phase, a, b in windows if a <= start <= end <= b]
                    if matches:
                        row["phase"] = matches[0]
                        rows.append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append(dict(file=path.name, line=number, error_type=type(exc).__name__))
    with (run_dir / "server_processor_calls.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summaries = []
    for phase in ("warmup", "steady", "retry", "profile"):
        for kind in ("multimodal_apply", "hf_processor"):
            selected = [r for r in rows if r["phase"] == phase and r["kind"] == kind]
            summaries.append(dict(phase=phase, kind=kind, call_count=len(selected),
                failed_calls=sum(not r["success"] for r in selected),
                total_wall_time_ms=sum(r["wall_time_ms"] for r in selected) if selected else None,
                wall_time_ms=stats([r["wall_time_ms"] for r in selected]),
                thread_cpu_time_ms=stats([r["thread_cpu_time_ms"] for r in selected])))
    steady = [r for r in rows if r["phase"] == "steady" and r["kind"] in ("multimodal_apply", "hf_processor")]
    status = dict(status="measured" if steady else "unavailable", summaries=summaries,
        hook_metadata=metadata, parse_errors=errors,
        unavailable_reason=None if steady else "No actual steady processor calls observed: inspect bootstrap, interfaces and cache behavior; not zero cost",
        scope="multimodal_apply includes cache lookup/placeholder/input organization; hf_processor is nested actual HF call; never add both",
        request_attribution="request_id only when real InputProcessor context encloses the call; per-sample aggregation not enabled; no order/time guess",
        cache_scope="HF invocation observed separately; no HF call alone does not prove cache hit",
        concurrency_scope="call wall sums overlap at workers=8; not workload wall share; thread CPU excludes other threads")
    dump_json(run_dir / "server_processor_status.json", status)
    write_csv(run_dir / "server_processor_statistics.csv", summaries)
    return status
