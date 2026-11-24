"""Chrome trace analysis with correlation-based CUDA attribution, not CPU-range-as-GPU timing."""
from __future__ import annotations

import gzip
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from benchmark.metrics import overlap_duration, stats, union_duration, write_csv

STAGE_NAMES = ("vision_encoder", "projector", "prefill", "decode", "sampling", "tokenizer_decode")
STAGE_ALIASES = {
    "runner_prefill": "prefill",
    "runner_decode": "decode",
}


def arg_value(event, *keys):
    args = event.get("args", {})
    for key in keys:
        if key in args:
            return args[key]
    return None


def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("traceEvents", []) if isinstance(data, dict) else data


def analyze_events(events, small_kernel_us=10):
    cpu = [dict(e) for e in events if e.get("ph") == "X" and e.get("dur", 0) >= 0
           and e.get("cat") in ("cpu_op", "user_annotation", "cpu_annotation")]
    runtime = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    copies = [e for e in events if e.get("ph") == "X" and "memcpy" in e.get("cat", "").lower()]
    threads = defaultdict(list)
    for index, event in enumerate(cpu):
        event["_id"] = index
        event["_children"] = []
        threads[(event.get("pid"), event.get("tid"))].append(event)
    external = {}
    for items in threads.values():
        stack = []
        for e in sorted(items, key=lambda e: (e["ts"], -e["dur"])):
            while stack and e["ts"] >= stack[-1]["ts"] + stack[-1]["dur"]:
                stack.pop()
            while stack and e["ts"] + e["dur"] > stack[-1]["ts"] + stack[-1]["dur"]:
                stack.pop()
            e["_parent"] = stack[-1] if stack else None
            if stack:
                stack[-1]["_children"].append((e["ts"], e["ts"] + e["dur"]))
            ext = arg_value(e, "External id", "External Id", "external id")
            if ext is not None:
                external[(e.get("pid"), str(ext))] = e
            stack.append(e)
    operators = {}
    for e in cpu:
        if e.get("cat") != "cpu_op":
            continue
        row = operators.setdefault(e["name"], {"operator_name": e["name"], "call_count": 0,
                 "self_cpu_time_total_ms": 0.0, "cpu_time_total_ms": 0.0,
                 "self_cuda_time_total_ms": 0.0, "cuda_time_total_ms": 0.0,
                 "input_shapes": [], "flops": None, "memory_bytes": None})
        row["call_count"] += 1
        row["cpu_time_total_ms"] += e["dur"] / 1000
        row["self_cpu_time_total_ms"] += max(0, e["dur"] - union_duration(e["_children"])) / 1000
        shapes = arg_value(e, "Input Dims", "Input shapes", "Input Shapes")
        if shapes is not None and shapes not in row["input_shapes"]:
            row["input_shapes"].append(shapes)
        flops = arg_value(e, "FLOPs", "FLOPS", "flops")
        if isinstance(flops, (int, float)):
            row["flops"] = (row["flops"] or 0) + flops
    # Kineto external IDs are process-local. Runtime correlation IDs connect launches to kernels.
    correlations = {}
    for event in runtime:
        ext = arg_value(event, "External id", "External Id", "external id")
        corr = arg_value(event, "correlation", "Correlation id")
        if ext is not None and corr is not None:
            correlations[(event.get("pid"), str(corr))] = external.get((event.get("pid"), str(ext)))
    external_by_id = defaultdict(list)
    for (_, ext), owner in external.items():
        external_by_id[ext].append(owner)
    correlations_by_id = defaultdict(list)
    for (_, corr), owner in correlations.items():
        if owner is not None:
            correlations_by_id[corr].append(owner)
    stage_us = defaultdict(float)
    correlated = 0
    operator_correlated = 0
    for kernel in kernels:
        ext = arg_value(kernel, "External id", "External Id", "external id")
        owner = external.get((kernel.get("pid"), str(ext)))
        if owner is None:
            # CUDA events often use device PID, unlike host events. Resolve only unique matches.
            possible = external_by_id.get(str(ext), []) if ext is not None else []
            owner = possible[0] if len(possible) == 1 else None
        if owner is None:
            corr = arg_value(kernel, "correlation", "Correlation id")
            possible = correlations_by_id.get(str(corr), []) if corr is not None else []
            owner = possible[0] if len(possible) == 1 else None
        if owner is None:
            continue
        correlated += 1
        chain = []
        e = owner
        while e is not None:
            chain.append(e)
            e = e.get("_parent")
        # Attribute kernel self time to nearest CPU op; inclusive time to each ancestor.
        nearest = next((e for e in chain if e.get("cat") == "cpu_op"), None)
        if nearest is not None:
            operator_correlated += 1
            operators[nearest["name"]]["self_cuda_time_total_ms"] += kernel["dur"] / 1000
        for e in chain:
            if e.get("cat") == "cpu_op":
                operators[e["name"]]["cuda_time_total_ms"] += kernel["dur"] / 1000
        # Deepest stage wins: projector nested inside vision is not counted twice.
        label = next((STAGE_ALIASES.get(e["name"].split("::", 1)[1], e["name"].split("::", 1)[1])
                      for e in chain
                      if e.get("name", "").startswith("benchmark::")
                      and (e["name"].split("::", 1)[1] in STAGE_NAMES
                           or e["name"].split("::", 1)[1] in STAGE_ALIASES)), None)
        if label:
            stage_us[label] += kernel["dur"]
    if not operator_correlated:
        for row in operators.values():
            row["self_cuda_time_total_ms"] = None
            row["cuda_time_total_ms"] = None
    cpu_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in cpu]
    gpu_intervals = [(e["ts"], e["ts"] + e["dur"]) for e in kernels + copies]
    gpu_active = union_duration(gpu_intervals)
    span = max(b for _, b in gpu_intervals) - min(a for a, _ in gpu_intervals) if gpu_intervals else None
    durations = [e["dur"] for e in kernels]
    sync = [e for e in runtime if "Synchronize" in e.get("name", "")]
    launch = [e for e in runtime if "LaunchKernel" in e.get("name", "") or "GraphLaunch" in e.get("name", "")]
    transfers = {}
    for direction in ("HtoD", "DtoH"):
        selected = [e for e in copies if direction.lower() in e.get("name", "").lower()]
        sizes = [arg_value(e, "bytes", "Bytes", "size") for e in selected]
        transfers[direction] = {"count": len(selected),
                                "time_ms": sum(e["dur"] for e in selected) / 1000 if selected else None,
                                "bytes": sum(sizes) if sizes and all(isinstance(v, (int, float)) for v in sizes) else None}
    allocations = [e for e in events if e.get("name") == "[memory]"]
    thread_index = {}
    for key, items in threads.items():
        ordered = sorted(items, key=lambda e: (e["ts"], -e["dur"]))
        thread_index[key] = ([e["ts"] for e in ordered], ordered)
    for allocation in allocations:
        ext = arg_value(allocation, "External id", "External Id")
        owner = external.get((allocation.get("pid"), str(ext)))
        if owner is None:
            starts, ordered = thread_index.get((allocation.get("pid"), allocation.get("tid")), ([], []))
            index = bisect_right(starts, allocation.get("ts", -1)) - 1
            owner = ordered[index] if index >= 0 else None
            while owner is not None and not owner["ts"] <= allocation["ts"] <= owner["ts"] + owner["dur"]:
                owner = owner.get("_parent")
        while owner is not None and owner.get("cat") != "cpu_op":
            owner = owner.get("_parent")
        size = arg_value(allocation, "Bytes", "bytes")
        if owner is not None and isinstance(size, (int, float)):
            row = operators[owner["name"]]
            row["memory_bytes"] = (row["memory_bytes"] or 0) + size
    cuda_allocations = [e for e in allocations if arg_value(e, "Device Type", "Device type") in (1, "CUDA", "cuda")]
    allocated = [arg_value(e, "Total Allocated") for e in cuda_allocations]
    reserved = [arg_value(e, "Total Reserved") for e in cuda_allocations]
    allocated = [v for v in allocated if isinstance(v, (int, float))]
    reserved = [v for v in reserved if isinstance(v, (int, float))]
    observed_stages = {e.get("name", "").split("::", 1)[1] for e in cpu if e.get("name", "").startswith("benchmark::")}
    result = {"source": "representative_worker_chrome_trace", "time_unit": "ms",
              "kernel_count": len(kernels) if gpu_intervals else None, "kernel_duration_us": stats(durations),
              "small_kernel_threshold_us": small_kernel_us,
              "small_kernel_count": sum(d < small_kernel_us for d in durations),
              "small_kernel_ratio": sum(d < small_kernel_us for d in durations) / len(kernels) if kernels else None,
              "kernel_launch_host_time_ms": sum(e["dur"] for e in launch) / 1000 if launch else None,
              "kernel_launch_count": len(launch),
              "gpu_active_time_ms": gpu_active / 1000 if gpu_intervals else None,
              "gpu_idle_time_ms": (span - gpu_active) / 1000 if span is not None else None,
              "gpu_activity_window_ms": span / 1000 if span is not None else None,
              "gpu_activity_scope": "first through last observed GPU event; excludes leading/trailing request idle",
              "cpu_gpu_overlap_ratio": overlap_duration(cpu_intervals, gpu_intervals) / gpu_active if gpu_active else None,
              "cuda_sync_time_ms": sum(e["dur"] for e in sync) / 1000 if runtime else None,
              "cuda_sync_count": len(sync) if runtime else None,
              "cuda_memcpy_async_host_time_ms": sum(e["dur"] for e in runtime if "MemcpyAsync" in e.get("name", "")) / 1000 if runtime else None,
              "transfers": transfers,
              "aten_to_count": sum(e["name"] == "aten::to" for e in cpu) if cpu else None,
              "aten_cpu_count": sum(e["name"] == "aten::cpu" for e in cpu) if cpu else None,
              "aten_item_count": sum(e["name"] == "aten::item" for e in cpu) if cpu else None,
              "aten_copy_count": sum(e["name"] == "aten::copy_" for e in cpu) if cpu else None,
              "correlated_kernel_count": correlated,
              "cuda_operator_attributed_kernel_count": operator_correlated,
              "kernel_attribution_ratio": correlated / len(kernels) if kernels else None,
              "stage_kernel_time_ms": {stage: stage_us[stage] / 1000 if stage in stage_us else None for stage in STAGE_NAMES},
              "stage_time_semantics": "sum of correlated GPU kernel durations; exclusive deepest-stage attribution, not stage wall latency",
              "observed_stage_ranges": sorted(observed_stages),
              "unattributed_kernel_time_ms": (sum(durations) - sum(stage_us.values())) / 1000 if kernels else None,
              "stage_attribution_note": "runner_prefill/runner_decode aliases are normalized to prefill/decode; stage kernel work remains correlation-limited",
              "trace_peak_allocated_bytes": max(allocated) if allocated else None,
              "trace_peak_reserved_bytes": max(reserved) if reserved else None,
              "profile_memory_events": len(allocations),
              "cuda_graph_launch_count": sum("GraphLaunch" in e.get("name", "") for e in runtime),
              "unavailable_reason": "absent Python stage ranges under CUDA Graph replay are not inferred from host TTFT"}
    return result, sorted(operators.values(), key=lambda r: r["self_cuda_time_total_ms"] or 0, reverse=True)


def _stats_cache_path(run_dir, trace_path):
    return run_dir / "trace_stats" / (trace_path.name + ".stats.json.gz")


def _analyze_trace(run_dir, path, small_kernel_us):
    """Return (data, operators, kernels); raw trace first, else stats extracted before it was pruned."""
    if path.is_file():
        events = load_trace(path)
        data, operators = analyze_events(events, small_kernel_us)
        kernels = defaultdict(list)
        for event in events:
            if event.get("ph") == "X" and event.get("cat") == "kernel":
                kernels[event["name"]].append(event["dur"])
        return data, operators, dict(kernels)
    cache = _stats_cache_path(run_dir, path)
    if not cache.is_file():
        raise OSError(f"trace missing and no extracted statistics: {path}")
    with gzip.open(cache, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"], payload["operators"], payload["kernels"]


def select_retained_traces(profile_records, keep, seed):
    """Seeded random sample IDs whose raw traces are kept; None keeps every trace."""
    if keep is None:
        return None
    import random
    ids = sorted({r["sample_id"] for r in profile_records if r.get("trace_files")})
    return set(random.Random(seed).sample(ids, min(keep, len(ids))))


def collect_traces(run_dir, profile_records, small_kernel_us=10, keep_raw=None, seed=42):
    # keep_raw: number of profiled samples whose raw traces stay on disk; the rest are reduced to
    # extracted statistics (trace_stats/) after analysis. None keeps all raw traces.
    results = []
    all_operators = defaultdict(list)
    kernel_durations = defaultdict(list)
    representative = None
    retained = select_retained_traces(profile_records, keep_raw, seed)
    for record in profile_records:
        # Analyze each worker trace independently: external IDs are not unique across processes/files.
        sample_results = []
        keep_this = retained is None or record["sample_id"] in retained
        if retained is not None and record.get("trace_files"):
            record["raw_trace_retained"] = keep_this
        for file in record.get("trace_files", []):
            path = Path(file)
            if not path.is_absolute():
                path = run_dir / path
            if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
                continue
            try:
                data, operators, kernels = _analyze_trace(run_dir, path, small_kernel_us)
                if not keep_this and path.is_file():
                    cache = _stats_cache_path(run_dir, path)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(cache, "wt", encoding="utf-8") as f:
                        json.dump({"trace_file": str(file), "data": data, "operators": operators,
                                   "kernels": kernels}, f)
                    path.unlink()
                for name, durations in kernels.items():
                    kernel_durations[name].extend(durations)
                if not data["kernel_count"] and not operators:
                    continue
                data.update(sample_id=record["sample_id"], trace_file=str(file), phase="profile",
                            trace_covers_full_request=record.get("trace_covers_full_request", False),
                            profile_trace_window=record.get("profile_trace_window"))
                results.append(data)
                sample_results.append(data)
                for row in operators:
                    all_operators[row["operator_name"]].append(row)
                if representative is None and data["kernel_count"] and path.is_file():
                    representative = str(file)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                results.append({"sample_id": record["sample_id"], "trace_file": str(path), "error": str(exc)})
        record["trace_analysis"] = sample_results
    merged = []
    for name, rows in all_operators.items():
        item = {"operator_name": name}
        for field in ("call_count", "self_cpu_time_total_ms", "cpu_time_total_ms", "self_cuda_time_total_ms", "cuda_time_total_ms", "flops", "memory_bytes"):
            values = [r.get(field) for r in rows]
            item[field] = sum(v for v in values if v is not None) if any(v is not None for v in values) else None
        item["input_shapes"] = [r["input_shapes"] for r in rows]
        merged.append(item)
    merged.sort(key=lambda r: r.get("self_cuda_time_total_ms") or 0, reverse=True)
    write_csv(run_dir / "operator_top50.csv", merged[:50], fields=["operator_name", "call_count",
              "self_cpu_time_total_ms", "cpu_time_total_ms", "self_cuda_time_total_ms", "cuda_time_total_ms", "flops", "memory_bytes", "input_shapes"])
    kernel_rows = [{"kernel_name": name, "total_duration_us": sum(values), **stats(values)}
                   for name, values in kernel_durations.items()]
    kernel_rows.sort(key=lambda r: r["total_duration_us"], reverse=True)
    write_csv(run_dir / "kernel_top50.csv", kernel_rows[:50], fields=["kernel_name", "total_duration_us", "count", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"])
    write_csv(run_dir / "kernel_statistics.csv", results)
    transfer_rows = []
    for r in results:
        transfer_rows.append({k: r.get(k) for k in ("sample_id", "trace_file", "cuda_sync_time_ms", "cuda_sync_count",
                              "cuda_memcpy_async_host_time_ms", "cpu_gpu_overlap_ratio", "aten_to_count", "aten_cpu_count", "aten_item_count", "aten_copy_count", "transfers", "error")})
    write_csv(run_dir / "cpu_gpu_transfer.csv", transfer_rows)
    if representative is not None:
        # Reference the original trace; never serialize another full payload.
        (run_dir / "trace_representative.ref.json").write_text(json.dumps({
            "trace_file": representative, "relative_to": "dataset run directory",
            "format": "chrome_trace", "storage": "original worker trace, not a copy"}, indent=2), encoding="utf-8")
    return results
