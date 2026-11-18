#!/usr/bin/env python3
"""Aggregate only observed request records, never directory file counts or word proxies."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.metrics import (PADDING_FIELDS, STAGES, TOKEN_FIELDS, area_bucket, dump_json,
                               exact_sum, length_bucket, load_jsonl, output_bucket, ratio, stats, write_csv)
from benchmark.trace_analysis import collect_traces

PROM_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+\S+)?$')


def read_prometheus(text):
    result = {}
    for line in text.splitlines():
        match = PROM_RE.match(line)
        if match:
            try:
                value = float(match[3])
                if value == value and abs(value) != float("inf"):
                    result[(match[1], match[2] or "")] = value
            except ValueError:
                continue
    return result


def prometheus_summary(run_dir, samples, start, end):
    before_path, after_path = run_dir / "metrics_before.prom", run_dir / "metrics_after.prom"
    before = read_prometheus(before_path.read_text(encoding="utf-8")) if before_path.exists() else {}
    after = read_prometheus(after_path.read_text(encoding="utf-8")) if after_path.exists() else {}
    series = defaultdict(list)
    for sample in samples:
        if sample.get("phase") != "steady" or sample.get("phase_end", sample.get("phase")) != "steady" or not (
                start <= sample["monotonic_s"] <= sample.get("sample_end_monotonic_s", sample["monotonic_s"]) <= end):
            continue
        path = run_dir / "monitor" / sample.get("prometheus_file", "missing")
        if path.is_file():
            for key, value in read_prometheus(path.read_text(encoding="utf-8")).items():
                series[key].append(value)
    rows = []
    for key in sorted(set(before) | set(after) | set(series)):
        name, labels = key
        is_counter = name.endswith(("_total", "_sum", "_count", "_bucket"))
        reset = key in before and key in after and after[key] < before[key] if is_counter else False
        delta = after[key] - before[key] if is_counter and key in before and key in after and not reset else None
        rows.append({"metric": name, "labels": labels, "before": before.get(key), "after": after.get(key),
                     "delta": delta, "counter_reset": reset, "sample_statistics": stats(series[key])})
    write_csv(run_dir / "vllm_metrics.csv", rows)
    histograms = {}
    for row in rows:
        if not row["metric"].endswith("_sum"):
            continue
        count_name = row["metric"][:-4] + "_count"
        count = next((r for r in rows if r["metric"] == count_name and r["labels"] == row["labels"]), {})
        if row["delta"] is not None and count.get("delta"):
            histograms[row["metric"] + row["labels"]] = {"sum": row["delta"], "count": count["delta"],
                                                       "mean": row["delta"] / count["delta"],
                                                       "scope": "server_histogram_delta_including_all_requests_in_window"}
    return rows, histograms


def throughput(records, elapsed):
    good = [r for r in records if r.get("success")]
    completed = [r for r in records if r.get("finish_reason") is not None]
    generated = exact_sum(completed, "generated_tokens")
    prompt = exact_sum(completed, "total_input_tokens")
    total = exact_sum(completed, "total_processed_tokens")
    docs_known = bool(records) and all(r.get("document_id") is not None for r in records)
    documents = defaultdict(list)
    if docs_known:
        for record in records:
            documents[record["document_id"]].append(record)
    complete_docs = sum(all(r.get("success") for r in pages) for pages in documents.values())
    return {"elapsed_s": elapsed, "attempted_requests": len(records), "successful_requests": len(good),
            "completed_generation_requests": len(completed), "completed_requests_per_second": ratio(len(completed), elapsed),
            "requests_per_second": ratio(len(good), elapsed),
            "documents_per_second": ratio(complete_docs, elapsed) if docs_known else None,
            "pages_per_second": ratio(sum(r.get("num_pages", 0) for r in good), elapsed),
            "images_per_second": ratio(sum(r.get("num_images", 0) for r in good), elapsed),
            "prompt_tokens_per_second": ratio(prompt, elapsed), "generated_tokens_per_second": ratio(generated, elapsed),
            "total_tokens_per_second": ratio(total, elapsed),
            "observed_generated_tokens": sum(r.get("generated_tokens") or 0 for r in records),
            "generation_token_coverage": ratio(sum(r.get("generated_tokens") is not None for r in records), len(records)),
            "prefill_tokens_per_second": None, "decode_tokens_per_second": None,
            "useful_output_tokens_per_second": ratio(exact_sum(good, "useful_output_tokens"), elapsed), "model_only_throughput": None,
            "useful_output_tokens_definition": "tokenizer count of accepted postprocessed text; not raw generated-ID count",
            "scope": "wall-clock full workload, including client I/O/postprocess; warmup, retries and profiler excluded",
            "unavailable_reasons": {"documents_per_second": "document_id annotations required; pages are not assumed distinct documents",
                                    "model_only_throughput": "requires actual execution-time attribution under overlapping continuous batching",
                                    "useful_output_tokens_per_second": "postprocessed text is not an actual generated-token subset"}}


def length_imbalance(records):
    output = {}
    for name, field in (("prompt", "prompt_text_tokens"), ("visual", "visual_tokens"),
                        ("output", "generated_tokens"), ("total_input", "total_input_tokens")):
        values = stats([r.get(field) for r in records])
        output[name] = {"field": field, "statistics": values,
                        "length_imbalance_ratio": ratio(values["max"], values["mean"]),
                        "max_to_mean_length": ratio(values["max"], values["mean"]),
                        "p95_to_median_length": ratio(values["p95"], values["p50"])}
    return output


def buckets(records, elapsed, field, bucket_fn):
    groups = defaultdict(list)
    for r in records:
        groups[bucket_fn(r.get(field))].append(r)
    result = []
    for name, members in sorted(groups.items()):
        good = [r for r in members if r.get("success")]
        serial_service = sum(r["end_to_end_latency_ms"] for r in members) / 1000
        result.append({"bucket": name, "sample_count": len(members), "successful_samples": len(good),
                       "mean_visual_tokens": stats([r.get("visual_tokens") for r in members])["mean"],
                       "ttft_ms": stats([r.get("ttft_ms") for r in members]),
                       "prefill_time_ms": stats([r.get("prefill_time_ms") for r in members]),
                       "decode_time_ms": stats([r.get("decode_time_ms") for r in members]),
                       "prefill_kernel_work_ms": stats([r.get("prefill_kernel_work_ms") for r in members]),
                       "decode_kernel_work_ms": stats([r.get("decode_kernel_work_ms") for r in members]),
                       "vision_encoder_kernel_work_ms": stats([r.get("vision_encoder_kernel_work_ms") for r in members]),
                       "latency_ms": stats([r.get("end_to_end_latency_ms") for r in members]),
                       "peak_allocated_memory_gb": stats([r.get("peak_allocated_memory_gb") for r in members]),
                       "pages_per_second": ratio(sum(r.get("num_pages", 0) for r in good), elapsed),
                       "output_tokens_per_second": ratio(exact_sum(members, "generated_tokens"), elapsed),
                       "throughput_scope": "bucket contribution to shared workload wall time, not isolated bucket throughput",
                       "serial_service_pages_per_second": ratio(len(good), serial_service)})
    return result


def worker_values(snapshot):
    return snapshot.get("results", []) if snapshot.get("status") == "measured" else []


def coverage_inventory(records, summary, profiles, traces, quality):
    """An explicit inventory includes unavailable measurements, with source and reason."""
    rows = []
    def add(metric, section, values, source, reason, output):
        count = sum(v is not None and (not isinstance(v, dict) or "count" not in v or v["count"] > 0) for v in values)
        status = "measured" if count else "unavailable"
        if metric == "requested_max_tokens" and records and all(r.get("requested_max_tokens") is None for r in records):
            status = "omitted_by_policy"
        rows.append({"metric": metric, "design_section": section, "observations": count,
                     "status": status, "source": source,
                     "unavailable_reason": reason if not count else None, "output": output})
    for field in ("server_version", "dtype", "cuda_graph_mode", "attention_backend", "compilation_config",
                  "cache_config", "scheduler_config", "max_model_len", "model_commit_hash", "tokenizer_revision", "kv_cache_num_gpu_blocks", "kv_cache_block_size"):
        values = summary["runtime"].get(field)
        values = values if isinstance(values, list) else [values]
        values = [v for v in values if v is not None and str(v) not in ("unknown", "None", "[]", "")]
        add("vllm_version" if field == "server_version" else field, "2,4", values,
            "actual server version endpoint or worker config", "installed runtime does not expose verified value", "config.json,summary.json")
    for field in ("running_requests", "waiting_requests", "kv_cache_usage_ratio", "preemptions", "prefix_cache_hits", "prefix_cache_queries"):
        values = [r.get("delta") if r.get("delta") is not None else r.get("after") for r in summary.get("scheduler_cache", {}).get(field, [])]
        add(field, "4", values, "version-matched raw Prometheus series; per-label snapshots and distributions",
            "metric not exposed by installed server", "vllm_metrics.csv,summary.json")
    add("prefix_cache_hit_rate", "4", [summary.get("scheduler_cache", {}).get("prefix_cache_hit_rate")],
        "delta cached tokens / delta queried tokens", "no exposed prefix-cache counters or no queries", "summary.json")
    queue = [v["mean"] for k, v in summary["server_histogram_deltas"].items() if "request_queue_time_seconds_sum" in k]
    add("queue_time_seconds", "4", queue, "actual request queue histogram delta mean", "queue histogram unavailable", "vllm_metrics.csv")
    for field in TOKEN_FIELDS:
        add(field, "6", [r.get(field) for r in records],
            "server usage" if field in ("generated_tokens", "total_input_tokens", "total_processed_tokens") else "separate processor audit (not server forward)",
            "not exposed or not semantically verified by installed processor/server", "token_statistics.csv")
    for name in ("mean", "std", "min", "max", "p50", "p90", "p95", "p99"):
        add("output_token_" + name, "6", [summary["tokens"]["generated_tokens"][name]],
            "actual usage distribution", "actual completion usage unavailable", "token_statistics.csv")
    for field in ("requested_max_tokens", "effective_max_tokens", "actual_generated_tokens", "finish_reason"):
        add(field, "2,3", [r.get(field) for r in records], "request payload, installed server limit helper, actual response",
            "omitted request max_tokens is intentionally null; missing effective helper is not replaced by a guessed cap", "per_sample_metrics.jsonl")
    for kind in ("multimodal_apply", "hf_processor"):
        measurements = [r["wall_time_ms"]["mean"] for r in summary["server_processor"]["summaries"]
                        if r["phase"] == "steady" and r["kind"] == kind]
        add("server_" + kind + "_time_ms", "5,8", measurements, "actual server processor method calls",
            summary["server_processor"]["unavailable_reason"] or "actual HF call not observed; no cache inference",
            "server_processor_statistics.csv")
    for field in ("eos_stop_rate", "length_stop_rate", "truncation_rate", "empty_output_rate", "error_rate"):
        add(field, "3", [summary["stop_statistics"].get(field)], "completed/attempted request populations explicitly separated",
            "terminal token evidence or completed population unavailable", "summary.json")
    for name in ("cold_start_latency_s", "model_load_time_s", "cuda_initialization_time_s"):
        add(name, "8", [summary["startup"].get(name)], "server launch readiness or explicit startup log timing",
            "phase not exposed in startup logs; readiness is not pure model load/CUDA init", "summary.json")
    for dimension in ("prompt", "visual", "output", "total_input"):
        for name in ("length_imbalance_ratio", "max_to_mean_length", "p95_to_median_length"):
            add(dimension + "_" + name, "7", [summary["length_imbalance"][dimension].get(name)],
                "actual length population distribution, NOT batch padding", "input/output lengths missing", "summary.json")
    for field in ("file_io_cpu_time_ms", "image_decode_cpu_time_ms", "preprocessing_cpu_time_ms", "postprocessing_cpu_time_ms", "tokenizer_cpu_time_ms"):
        add(field, "10", [r.get(field) for r in records], "client thread_time clock; tokenizer worker CPU needs separate evidence",
            "CPU stage not observed; wall-clock duration is not CPU time", "per_sample_metrics.jsonl")
    for field in PADDING_FIELDS:
        add(field, "7", [r.get(field) for r in records], "actual processor/server observations only",
            "scheduler batch membership/physical image padding not observed; length imbalance is not padding", "padding_statistics.csv")
    for stage in STAGES:
        add(stage + "_time_ms", "5,8", [r.get(stage + "_time_ms") for r in records + profiles], "client steady observations or representative CUDA Event spans, phase-separated",
            "not measurable at API client and no supported worker phase evidence; inspect trace/histogram separately", "stage_latency.csv")
    for field in ("end_to_end_latency_ms", "request_latency_ms", "ttft_ms", "ttft_e2e_ms", "tpot_ms", "gpu_ttft_ms"):
        add(field, "8,17", [r.get(field) for r in records + profiles], "host request/stream timestamps or representative-only CUDA Event evidence, phase-separated",
            "no direct measurement; host intervals do not replace GPU phase times", "stage_latency.csv")
    add("itl_ms", "8,17", [r.get("itl_ms") for r in records], "one logprob token per SSE event only",
        "coalesced/unverified text chunks do not yield token ITL", "summary.json")
    for field, value in summary["throughput"].items():
        if field.endswith("per_second") or field == "model_only_throughput":
            add(field, "9", [value], "wall-clock workload unless explicitly model-only",
                "required input or execution/quality attribution not available", "throughput_statistics.csv")
    for field, value in summary["memory"].items():
        add(field, "10", [value] if not isinstance(value, str) else [], "worker run-wide allocator snapshot",
            "not available from installed worker; never derived as allocated minus weights", "memory_statistics.csv")
    for field in ("gpu_utilization_percent", "sm_utilization_percent", "tensor_core_utilization_percent",
                  "gpu_memory_utilization_percent", "gpu_power_w", "gpu_temperature_c", "gpu_sm_clock_mhz",
                  "server_process_cpu_percent", "client_process_cpu_percent", "client_process_rss_bytes", "system_cpu_percent", "server_process_rss_bytes", "windows_host_memory_bytes"):
        add(field, "10", [summary["system"].get(field, {}).get("mean")], "nvidia-smi, Linux /proc, optional local DCGM exporter",
            "not supplied by nvidia-smi/proc; SM/Tensor Core require DCGM/Nsight, Windows memory requires host telemetry", "summary.json")
    for field in ("kernel_count", "kernel_duration_us", "small_kernel_ratio", "kernel_launch_host_time_ms",
                  "gpu_active_time_ms", "gpu_idle_time_ms", "cuda_sync_time_ms", "cuda_sync_count",
                  "cuda_memcpy_async_host_time_ms", "cpu_gpu_overlap_ratio", "aten_to_count", "aten_cpu_count", "aten_item_count", "aten_copy_count"):
        add(field, "10,11", [r.get(field) for r in traces], "representative worker Chrome trace only",
            "worker CUDA trace missing or relevant events not captured", "kernel_statistics.csv,cpu_gpu_transfer.csv")
    for field in ("HtoD", "DtoH"):
        for subfield in ("bytes", "time_ms"):
            add(field + "_" + subfield, "10", [r.get("transfers", {}).get(field, {}).get(subfield) for r in traces],
                "actual GPU memcpy trace events", "transfer direction/size not exposed in captured trace", "cpu_gpu_transfer.csv")
    for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling", "tokenizer_decode"):
        add(stage + "_profile_kernel_time_ms", "5,11", [r.get("stage_kernel_time_ms", {}).get(stage) for r in traces],
            "correlated kernel sum, exclusive deepest worker stage range; not wall latency",
            "installed model lacks stage annotation or CUDA Graph replay bypasses Python hooks", "kernel_statistics.csv")
    for field in ("original_width", "original_height", "original_pixels", "image_megapixels", "resized_width",
                  "resized_height", "resized_pixels", "image_grid_height", "image_grid_width", "pixel_values_shape"):
        add(field, "12", [r.get(field) for r in records], "image decode or processor audit",
            "processor actual resize geometry unavailable; client resize is recorded separately", "per_sample_metrics.jsonl")
    for field in quality.get("required_metrics", []):
        metric_value = quality.get("metrics", {}).get(field)
        if field in quality.get("missing_metrics", []):
            metric_value = None
        add(field, "13", [metric_value], quality.get("source", "external evaluator"),
            "quality evaluator/annotations/version missing or metric not supplied", "quality.json")
    for field, value in quality.get("metrics", {}).items():
        if field not in quality.get("required_metrics", []):
            add(field, "13", [value], quality.get("source", "external evaluator"), "quality metric unavailable", "quality.json")
    return rows


def acceptance(summary, records, traces, quality, inventory, profiles=()):
    memory = summary["memory"]
    tokens = summary["tokens"]
    present = lambda field: tokens.get(field, {}).get("count", 0) == len(records) and bool(records)
    required_stages = ("vision_encoder", "prefill", "decode")
    # Require one complete request, not disjoint samples or an initial trace window.
    trace_stages_complete = any(t.get("trace_covers_full_request") and
        all(t.get("stage_kernel_time_ms", {}).get(stage) is not None for stage in required_stages)
        for t in traces)
    event_stages_complete = any(r.get("success") and
        all(r.get(stage + "_time_ms") is not None for stage in required_stages)
        for r in profiles)
    stages_complete = trace_stages_complete or event_stages_complete
    good_traces = [t for t in traces if t.get("kernel_count")]
    criteria = [
        ("1_stage_dominance", stages_complete, "Vision/Prefill/Decode coverage from one measurement kind: CUDA Event spans or correlated kernel work, not host TTFT"),
        ("2_input_output_tokens", all(present(f) for f in ("prompt_text_tokens", "visual_tokens", "generated_tokens")), "full-sample text/vision/output counts; processor scope disclosed"),
        ("3_output_length_and_truncation", present("generated_tokens") and summary["stop_statistics"]["finish_reason_coverage"] == 1, "actual usage and finish_reason coverage"),
        ("4_cold_warm_steady", summary["startup"].get("cold_start_latency_s") is not None and summary["warmup_latency_ms"]["count"] >= 3, "cold server readiness, request warmup, steady kept separate"),
        ("5_steady_percentiles", summary["latency_ms"]["count"] == len(records), "request record population and latency percentiles"),
        ("6_throughput", all(summary["throughput"].get(k) is not None for k in ("pages_per_second", "images_per_second", "prefill_tokens_per_second", "generated_tokens_per_second")), "unit-specific wall/model throughput"),
        ("7_actual_padding", all(any(r.get(f) is not None for r in records) for f in ("prompt_padding_tokens", "visual_padding_tokens", "pixel_padding_area")), "actual processor/visual padding; null is not zero"),
        ("8_size_length_buckets", any(r.get("image_megapixels") is not None for r in records) and present("generated_tokens"), "area/output buckets with latency percentiles"),
        ("9_gpu_bottleneck_evidence", bool(good_traces) and stages_complete and summary["system"].get("gpu_utilization_percent", {}).get("count", 0) > 0, "timeline and stage evidence, not utilization-only diagnosis"),
        ("10_memory", all(memory.get(f) is not None for f in ("peak_allocated_bytes", "kv_cache_memory_bytes", "cpu_rss_peak_bytes")), "run-wide GPU peaks, actual KV allocations, CPU RSS"),
        ("11_transfer_sync", any(t.get("cuda_sync_count") is not None and t.get("aten_to_count") is not None for t in traces), "worker CPU/CUDA traces with runtime events"),
        ("12_operator_kernel_hotspots", any(t.get("cuda_operator_attributed_kernel_count", 0) > 0 for t in traces), "correlation-derived CUDA operator table and kernel statistics"),
        ("13_quality", quality.get("baseline_evaluation", {}).get("status") == "pass", "official/task quality coverage and explicitly declared baseline targets met"),
    ]
    return {"all_minimum_criteria_met": all(ok for _, ok, _ in criteria),
            "criteria": [{"id": name, "status": "pass" if ok else "missing_evidence", "requirement": why} for name, ok, why in criteria],
            "measured_metric_count": sum(r["status"] == "measured" for r in inventory),
            "unavailable_metrics": [r["metric"] for r in inventory if r["status"] == "unavailable"],
            "important": "files/fields existing do not mean a metric was measured; inspect metric_coverage.csv and raw evidence"}


def collect_run(run_dir, small_kernel_us=10):
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = load_jsonl(run_dir / "per_sample_metrics.jsonl")
    # Unexpected duplicates must fail rather than silently inflate denominators.
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate steady-state sample IDs")
    expected = load_jsonl(run_dir / "manifest.jsonl")
    if set(ids) != {s["sample_id"] for s in expected}:
        raise ValueError("steady-state records do not match manifest; incomplete run")
    warmup_path, profile_path = run_dir / "warmup.json", run_dir / "profile_metrics.json"
    warmups = json.loads(warmup_path.read_text(encoding="utf-8")) if warmup_path.exists() else []
    profiles = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else []
    from benchmark.processor_results import collect_processor_calls
    processor_status = collect_processor_calls(run_dir, config, records, warmups, profiles)
    traces = collect_traces(run_dir, profiles, small_kernel_us)
    dump_json(run_dir / "profile_metrics.json", profiles)
    monitor_path = run_dir / "monitor" / "system_samples.jsonl"
    system_samples = load_jsonl(monitor_path) if monitor_path.exists() else []
    start, end = config["steady_start_s"], config["steady_end_s"]
    steady_system = [r for r in system_samples if r.get("phase") == "steady" and r.get("phase_end", r.get("phase")) == "steady"
                     and start <= r["monotonic_s"] <= r.get("sample_end_monotonic_s", r["monotonic_s"]) <= end]
    prom_rows, histograms = prometheus_summary(run_dir, system_samples, start, end)
    elapsed = config["steady_elapsed_s"]
    successful = [r for r in records if r.get("success")]
    completed = [r for r in records if r.get("finish_reason") is not None]
    finish = Counter(r.get("finish_reason") for r in completed)
    quality_path = run_dir / "quality.json"
    if quality_path.exists():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if not isinstance(quality, dict) or not isinstance(quality.get("metrics", {}), dict):
                raise ValueError("quality output must contain an object of numeric metrics")
        except (ValueError, OSError) as exc:
            quality = {"status": "failed", "reason": str(exc), "metrics": {}}
    else:
        from benchmark.quality import field_quality
        mapping = json.loads((run_dir / "prediction_mapping.json").read_text(encoding="utf-8"))
        try:
            quality = field_quality(expected, mapping) if any("ground_truth_fields" in s for s in expected) else {
                "status": "unavailable", "reason": "official/task annotations and quality adapter required", "metrics": {}}
        except (ValueError, OSError, re.error) as exc:
            quality = {"status": "failed", "reason": str(exc), "metrics": {}}
    dataset = expected[0].get("dataset") if expected else None
    required_quality = ("text_edit", "formula_cdm", "table_teds") if dataset == "omnidocbench" else ("field_accuracy", "precision", "recall", "f1", "cer", "wer")
    quality["required_metrics"] = list(required_quality)
    import math
    invalid = [k for k, v in quality.get("metrics", {}).items() if isinstance(v, float) and not math.isfinite(v)]
    for key in invalid:
        quality["metrics"][key] = None
    if invalid:
        quality.update(status="failed", invalid_nonfinite_metrics=invalid)
    quality["missing_metrics"] = [k for k in required_quality if not isinstance(quality.get("metrics", {}).get(k), (int, float))
                                  or isinstance(quality.get("metrics", {}).get(k), bool)
                                  or not math.isfinite(quality["metrics"][k])]
    quality["expected_samples"] = len(expected)
    quality["sample_coverage_complete"] = quality.get("evaluated_samples") == len(expected)
    quality["version_matches_design"] = quality.get("benchmark_version") == "1.5" if dataset == "omnidocbench" else None
    if quality.get("status") == "complete" and (quality["missing_metrics"] or not quality["sample_coverage_complete"]
                                                 or (dataset == "omnidocbench" and not quality["version_matches_design"])):
        quality["status"] = "incomplete"
    from benchmark.quality import evaluate_baseline
    quality["baseline_evaluation"] = evaluate_baseline(quality)
    dump_json(quality_path, quality)
    snapshots = worker_values(config.get("worker_memory_after", {}))
    before_snapshots = worker_values(config.get("worker_memory_before", {}))
    memory = {}
    for field in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "model_weight_memory_bytes", "activation_memory_bytes", "kv_cache_memory_bytes", "input_image_memory_bytes", "inactive_split_bytes", "pinned_memory_bytes"):
        vals = [s.get(field) for s in snapshots]
        memory[field] = sum(vals) if vals and all(v is not None for v in vals) else None
    memory["cpu_rss_peak_bytes"] = stats([r.get("server_process_rss_bytes") for r in steady_system])["max"]
    memory["inactive_split_to_reserved_ratio"] = ratio(memory.get("inactive_split_bytes"), memory.get("reserved_bytes"))
    memory["memory_fragmentation"] = None
    memory["scope"] = "run-wide workers, not per-request; allocated KV capacity is not live token occupancy"
    oom_before = {s.get("pid"): s.get("allocator_oom_count_cumulative") for s in before_snapshots}
    oom_deltas = [s["allocator_oom_count_cumulative"] - oom_before[s["pid"]] for s in snapshots
                  if s.get("allocator_oom_count_cumulative") is not None and oom_before.get(s.get("pid")) is not None]
    memory["allocator_oom_count_delta"] = sum(oom_deltas) if oom_deltas else None
    memory["http_oom_error_count"] = sum(r.get("oom_error", False) for r in records)
    system_fields = ("gpu_utilization_percent", "gpu_memory_utilization_percent", "gpu_memory_used_mib", "gpu_power_w", "gpu_temperature_c", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz", "server_process_cpu_percent", "client_process_cpu_percent", "client_process_rss_bytes", "system_cpu_percent", "server_process_rss_bytes", "sm_utilization_percent", "tensor_core_utilization_percent", "pinned_memory_bytes", "windows_host_memory_bytes")
    startup_path = run_dir.parent / "server_startup.json"
    startup = json.loads(startup_path.read_text(encoding="utf-8")) if startup_path.exists() else {}
    startup_log = run_dir.parent / "logs" / "vllm_server.log"
    if startup_log.is_file():
        text = startup_log.read_text(encoding="utf-8", errors="replace")
        load_matches = re.findall(r"Loading model weights took[^\n]*?([0-9.]+) seconds", text)
        if load_matches:
            startup["model_load_time_s"] = float(load_matches[-1])
            startup["model_load_time_source"] = "explicit model weights loading log; not whole server initialization"
    initial_workers = worker_values(config.get("worker_initial", {}))
    runtime = {field: [s.get(field) for s in initial_workers] for field in
               ("dtype", "max_model_len", "cuda_graph_mode", "attention_backend", "compilation_config",
                "cache_config", "scheduler_config", "model_commit_hash", "tokenizer_revision", "kv_cache_num_gpu_blocks", "kv_cache_block_size")}
    runtime["server_version"] = config.get("version", {}).get("version")
    summary = {"dataset": dataset, "runtime": runtime, "sample_count": len(records), "successful_samples": len(successful),
               "failed_samples": len(records) - len(successful), "startup": startup,
               "warmup_latency_ms": stats([r.get("end_to_end_latency_ms") for r in warmups]),
               "latency_ms": stats([r.get("end_to_end_latency_ms") for r in records]),
               "successful_latency_ms": stats([r.get("end_to_end_latency_ms") for r in successful]),
               "request_latency_ms": stats([r.get("request_latency_ms") for r in records]),
               "ttft_ms": stats([r.get("ttft_ms") for r in records]),
               "ttft_e2e_ms": stats([r.get("ttft_e2e_ms") for r in records]),
               "itl_ms": stats([v for r in records for v in (r.get("itl_ms") or [])]),
               "chunk_interval_ms": stats([v for r in records for v in r.get("chunk_interval_ms", [])]),
               "itl_steady_ms": stats([v for r in records for v in (r.get("itl_ms") or [])[1:]]),
               "throughput": throughput(records, elapsed),
               "tokens": {f: stats([r.get(f) for r in records]) for f in TOKEN_FIELDS},
               "length_imbalance": length_imbalance(records), "memory": memory,
               "system": {f: stats([r.get(f) for r in steady_system]) for f in system_fields},
               "server_histogram_deltas": histograms,
               "stop_statistics": {"finish_reasons": dict(finish), "completed_requests": len(completed),
                   "finish_reason_coverage": ratio(len(completed), len(records)),
                   "stop_rate": ratio(finish["stop"], len(completed)),
                   "eos_stop_rate": ratio(sum(r.get("stop_reason_is_eos") is True for r in completed), len(completed))
                       if completed and all(r.get("stop_reason_is_eos") is not None for r in completed) else None,
                   "eos_token_evidence_coverage": ratio(sum(r.get("stop_reason_is_eos") is not None for r in completed), len(completed)),
                   "eos_unavailable_reason": "OpenAI stop alone does not distinguish EOS; actual worker terminal token IDs required",
                   "length_stop_rate": ratio(finish["length"], len(completed)),
                   "truncation_rate": ratio(finish["length"], len(completed)),
                   "empty_output_rate": ratio(sum(r.get("empty_output", False) or r.get("postprocessed_empty_output", False) for r in records), len(records)),
                   "error_rate": ratio(len(records) - len(successful), len(records))},
               "high_resolution_threshold_mp": 8, "high_resolution_sample_rate": ratio(sum((r.get("image_megapixels") or 0) > 8 for r in records), len(records)),
               "long_document_threshold_pages": None, "long_document_rate": None,
               "long_document_unavailable_reason": "document page groups and a declared long-document threshold are required",
               "quality": quality, "server_processor": processor_status, "profile_summary": traces,
               "profile_stage_latency_ms": {stage: stats([r.get(stage + "_time_ms") for r in profiles]) for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling")},
               "profile_model_only_generated_tokens_per_second": stats([r.get("model_only_generated_tokens_per_second") for r in profiles]),
               "profile_gpu_ttft_ms": stats([r.get("gpu_ttft_ms") for r in profiles]),
               "profile_throughput_scope": "representative CUDA Event spans with profiler; never mixed into ordinary steady throughput",
               "notes": ["all latency distributions include mean/std/min/p50/p90/p95/p99/max",
                         "chunk intervals are not called token ITL without one-token event evidence",
                         "processor audit is a separate pass, not measured server processor execution",
                         "profiled traces, warmups and retries are not included in steady-state throughput"]}
    # Server histogram phases are reported as aggregate observations, never per-request stage values.
    phase_totals = {}
    for phase, needles in (("prefill", ("request_prefill_time_seconds_sum",)), ("decode", ("request_decode_time_seconds_sum",))):
        rows = [r for r in prom_rows if any(r["metric"].endswith(n) for n in needles)]
        phase_totals[phase] = sum(r["delta"] for r in rows) if rows and all(r["delta"] is not None for r in rows) else None
    if phase_totals["prefill"]:
        summary["throughput"]["prefill_tokens_per_second"] = ratio(exact_sum(completed, "total_input_tokens"), phase_totals["prefill"])
    if phase_totals["decode"]:
        summary["throughput"]["decode_tokens_per_second"] = ratio(exact_sum(completed, "generated_tokens"), phase_totals["decode"])
    cache_scheduler = {}
    for canonical, aliases in {
        "running_requests": ("vllm:num_requests_running",),
        "waiting_requests": ("vllm:num_requests_waiting",),
        "kv_cache_usage_ratio": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "preemptions": ("vllm:num_preemptions_total",),
        "prefix_cache_hits": ("vllm:prefix_cache_hits_total",),
        "prefix_cache_queries": ("vllm:prefix_cache_queries_total",),
    }.items():
        found = []
        for alias in aliases:
            found = [r for r in prom_rows if r["metric"] == alias]
            if found:
                break
        cache_scheduler[canonical] = found
    hits = [r.get("delta") for r in cache_scheduler["prefix_cache_hits"]]
    queries = [r.get("delta") for r in cache_scheduler["prefix_cache_queries"]]
    cache_scheduler["prefix_cache_hit_rate"] = ratio(sum(hits), sum(queries)) if hits and queries and all(v is not None for v in hits + queries) else None
    cache_scheduler["prefix_cache_hit_rate_unit"] = "cached prefix tokens / queried prefix tokens, not request hit ratio"
    summary["scheduler_cache"] = cache_scheduler
    summary["server_phase_total_seconds"] = phase_totals
    summary["server_phase_throughput_scope"] = "tokens / sum server request-phase durations; overlapping requests are not device-active wall time"
    peak_gb = ratio(memory.get("peak_allocated_bytes"), 1024 ** 3)
    summary["quality_derived"] = {metric: {"quality_per_second": ratio(value, elapsed),
                                            "quality_per_gb": ratio(value, peak_gb)}
                                  for metric, value in quality.get("metrics", {}).items()
                                  if metric in ("quality_score", "field_accuracy", "precision", "recall", "f1", "formula_cdm", "table_teds")
                                  and isinstance(value, (float, int))}
    summary["quality_derived_note"] = "only higher-is-better scores; lower-is-better CER/WER/Edit are not divided into misleading quality ratios"
    price = config.get("gpu_cost_per_hour")
    summary["cost_per_page"] = ratio(price * elapsed / 3600, len(successful)) if price is not None else None
    summary["cost_definition"] = "declared single reserved-GPU price/hour times steady wall time / accepted pages; excludes startup and profiler"
    documents = defaultdict(list)
    if records and all(r.get("document_id") is not None for r in records):
        for r in records:
            documents[r["document_id"]].append(r)
        threshold = config.get("long_document_pages", 10)
        summary["long_document_threshold_pages"] = threshold
        summary["long_document_rate"] = sum(len(pages) > threshold for pages in documents.values()) / len(documents)
        summary["long_document_scope"] = "document_id groups present in this manifest; not inferred full original documents"
        summary["throughput"]["documents_per_second"] = ratio(sum(all(p.get("success") for p in pages) for pages in documents.values()), elapsed)
    dump_json(run_dir / "summary.json", summary)
    latency_rows = []
    for field in [s + "_time_ms" for s in STAGES] + ["end_to_end_latency_ms", "request_latency_ms", "ttft_ms", "ttft_e2e_ms", "host_tpot_after_first_ms", "time_per_generated_token_ms", "processor_audit_time_ms", "file_io_cpu_time_ms", "image_decode_cpu_time_ms", "preprocessing_cpu_time_ms", "postprocessing_cpu_time_ms"]:
        latency_rows.append({"metric": field, "phase": "steady", "scope": "separate_processor_audit" if field == "processor_audit_time_ms" else "request_observation", **stats([r.get(field) for r in records])})
    for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling", "tokenizer_decode"):
        per_sample_work = []
        for profile in profiles:
            values = [t.get("stage_kernel_time_ms", {}).get(stage) for t in profile.get("trace_analysis", [])]
            present = [v for v in values if v is not None]
            per_sample_work.append(sum(present) if present else None)
        latency_rows.append({"metric": stage + "_kernel_work_ms", "phase": "profile",
                             "scope": "sum correlated GPU kernel time per representative; NOT wall latency",
                             **stats(per_sample_work)})
    for field in [stage + "_time_ms" for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling")] + ["gpu_ttft_ms", "tpot_ms", "runner_prefill_step_time_ms", "runner_decode_step_time_ms"]:
        latency_rows.append({"metric": field, "phase": "profile", "scope": "representative current-stream CUDA Events; step fields inclusive, model stages exclusive",
                             **stats([r.get(field) for r in profiles])})
    for direction in ("HtoD", "DtoH"):
        latency_rows.append({"metric": direction + "_transfer_time_ms", "phase": "profile",
                             "scope": "GPU memcpy event duration sum per trace",
                             **stats([t.get("transfers", {}).get(direction, {}).get("time_ms") for t in traces])})
    for row in processor_status["summaries"]:
        latency_rows.append({"metric": "server_" + row["kind"] + "_time_ms", "phase": row["phase"],
                             "scope": "actual server call; nested scopes not additive; no per-request attribution",
                             **row["wall_time_ms"]})
    write_csv(run_dir / "stage_latency.csv", latency_rows)
    write_csv(run_dir / "token_statistics.csv", [{"metric": f, **v} for f, v in summary["tokens"].items()])
    write_csv(run_dir / "padding_statistics.csv", [{"sample_id": r["sample_id"], **{f: r.get(f) for f in PADDING_FIELDS},
                                                     "processor_attention_padding_tokens": r.get("processor_attention_padding_tokens"),
                                                     "processor_attention_padding_ratio": r.get("processor_attention_padding_ratio"),
                                                     "scope": "processor audit and scheduler padding are separate"} for r in records])
    write_csv(run_dir / "throughput_statistics.csv", [summary["throughput"]])
    memory_rows = [{"phase": "steady", "sample_id": None, **memory}]
    for profile in profiles:
        values = worker_values(profile.get("worker_memory_after", {}))
        row = {"phase": "profile", "sample_id": profile["sample_id"],
               "scope": "single representative with profiler; not steady-state per-sample memory"}
        for field in memory:
            if field in ("scope", "cpu_rss_peak_bytes", "allocator_oom_count_delta", "http_oom_error_count", "inactive_split_to_reserved_ratio", "memory_fragmentation"):
                continue
            observations = [s.get(field) for s in values]
            row[field] = sum(observations) if observations and all(v is not None for v in observations) else None
        memory_rows.append(row)
    write_csv(run_dir / "memory_statistics.csv", memory_rows)
    dimensions = [("image_area", "image_megapixels", area_bucket), ("visual_tokens", "visual_tokens", length_bucket),
                  ("text_tokens", "prompt_text_tokens", length_bucket), ("output_tokens", "generated_tokens", output_bucket),
                  ("has_table", "has_table", lambda v: str(v) if v is not None else "unknown"),
                  ("has_formula", "has_formula", lambda v: str(v) if v is not None else "unknown"),
                  ("document_type", "document_type", lambda v: str(v) if v is not None else "unknown"),
                  ("difficulty", "difficulty", lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True) if v is not None else "unknown")]
    profile_observations = [r for r in profiles if r.get("end_to_end_latency_ms") is not None]
    for profile in profile_observations:
        for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling"):
            observed = [t.get("stage_kernel_time_ms", {}).get(stage) for t in profile.get("trace_analysis", [])]
            observed = [v for v in observed if v is not None]
            profile[stage + "_kernel_work_ms"] = sum(observed) if observed else None
    dump_json(run_dir / "profile_metrics.json", profiles)
    for name, field, fn in dimensions:
        write_csv(run_dir / f"bucket_by_{name}.csv", buckets(records, elapsed, field, fn))
        diagnostic_rows = buckets(profile_observations, 0, field, fn)
        for row in diagnostic_rows:
            row.update(phase="profile", pages_per_second=None, output_tokens_per_second=None,
                       serial_service_pages_per_second=None, throughput_scope="profile diagnostics; throughput deliberately excluded")
        write_csv(run_dir / f"bucket_by_{name}_profile.csv", diagnostic_rows)
    inventory = coverage_inventory(records, summary, profiles, traces, quality)
    write_csv(run_dir / "metric_coverage.csv", inventory)
    result = acceptance(summary, records, traces, quality, inventory, profiles)
    result["sample_size_note"] = "fewer than recommended 30 samples; report actual population" if len(records) < 30 else None
    dump_json(run_dir / "acceptance_report.json", result)
    from benchmark.report import write_report
    write_report(run_dir / "bottleneck_report.md", summary, records, traces, result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--small-kernel-us", type=float, default=10)
    args = parser.parse_args()
    if args.small_kernel_us <= 0:
        parser.error("small-kernel-us must be positive")
    collect_run(args.run_dir, args.small_kernel_us)


if __name__ == "__main__":
    main()
