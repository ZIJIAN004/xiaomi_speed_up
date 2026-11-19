"""Load one single-GPU engine and reuse it across all benchmark datasets."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from benchmark.metrics import dump_json
from benchmark.prepare_datasets import DATASETS, write_manifest
from benchmark.run_benchmark import main as run_dataset, parse_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("omnidocbench", "midv500", "midv-lait", "model", "output"):
        parser.add_argument(name, type=Path)
    inputs = parser.parse_args()
    if inputs.output.exists():
        parser.error("output must be a new directory")
    if not inputs.model.is_dir():
        parser.error("model must be an existing local directory")
    root = inputs.output.resolve()
    root.mkdir(parents=True)
    manifests = root / "manifests"
    manifests.mkdir()
    names = ("omnidocbench",) if os.environ.get("BENCHMARK_RUNTIME_SMOKE") == "1" else DATASETS
    summaries = []
    for name in names:
        annotations = os.environ.get({"omnidocbench": "OMNIDOC_ANNOTATIONS", "midv500": "MIDV500_ANNOTATIONS", "midv-lait": "MIDV_LAIT_ANNOTATIONS"}[name])
        summaries.append(write_manifest(name, getattr(inputs, name), manifests,
                                        Path(annotations) if annotations else None))
    dump_json(manifests / "dataset_manifest_summary.json", summaries)
    profiler = root / "profiler_raw"
    processor = root / "processor_raw"
    profiler.mkdir()
    processor.mkdir()
    os.environ["VLLM_BENCHMARK_PROCESSOR_DIR"] = str(processor)
    from benchmark.offline_backend import OfflineBackend
    common = ["--backend", "offline", "--model", str(inputs.model), "--workers", "8",
              "--gpu", os.environ.get("GPU_ID", "1"), "--server-pid", str(os.getpid()),
              "--profiler-dir", str(profiler), "--server-processor-dir", str(processor)]
    for env, flag, default in (("MODEL_NAME", "--served-model-name", "Xiaomi-OCR-0"),
                               ("GPU_MEM_UTIL", "--gpu-memory-utilization", "0.90"),
                               ("MAX_MODEL_LEN", "--max-model-len", "16384"), ("DTYPE", "--dtype", "auto"),
                               ("REQUEST_TIMEOUT", "--request-timeout", "600"), ("WARMUP", "--warmup", "5"),
                               ("REPRESENTATIVES", "--representatives", "50"),
                               ("PROFILE_MAX_ITERATIONS", "--profile-max-iterations", "64"),
                               ("OUTPUT_RETENTION", "--output-retention", "sampled"),
                               ("RETAIN_SAMPLES", "--retain-samples", "10"),
                               ("RETAIN_WORST", "--retain-worst", "10"),
                               ("RETENTION_SEED", "--retention-seed", "42")):
        common += [flag, os.environ.get(env, default)]
    for env, flag in (("MAX_TOKENS", "--max-tokens"), ("MODULE_MAP", "--module-map"),
                      ("DCGM_URL", "--dcgm-url"), ("GPU_COST_PER_HOUR", "--gpu-cost-per-hour")):
        if os.environ.get(env):
            common += [flag, os.environ[env]]
    if os.environ.get("TRUST_REMOTE_CODE") == "1":
        common.append("--trust-remote-code")
    if os.environ.get("RETRY_REPETITIVE", "1") == "1":
        common.append("--retry-repetitive")
    args_list = []
    for name in names:
        argv = common + ["--manifest", str(manifests / (name + ".jsonl")), "--output", str(root / name)]
        quality = os.environ.get({"omnidocbench": "OMNIDOC_QUALITY_COMMAND", "midv500": "MIDV500_QUALITY_COMMAND", "midv-lait": "MIDV_LAIT_QUALITY_COMMAND"}[name])
        if quality:
            argv += ["--quality-command", quality]
        args_list.append(parse_args(argv))
    started = time.time()
    with OfflineBackend(args_list[0]) as backend:
        ready = time.time()
        dump_json(root / "server_startup.json", {"status": "ready", "backend": "offline",
                  "server_start_wall_s": started, "server_ready_wall_s": ready,
                  "cold_start_latency_s": ready-started,
                  "cold_start_scope": "offline engine construction and initialization, not HTTP readiness",
                  "model_load_time_s": None, "cuda_initialization_time_s": None})
        snapshots = backend.rpc("benchmark_snapshot", ["false"])
        dump_json(root / "worker_startup.json", snapshots)
        if len(snapshots) != 1 or snapshots[0].get("peak_allocated_bytes") is None:
            raise RuntimeError("single-GPU worker instrumentation unavailable; inspect worker_startup.json")
        dump_json(root / "runtime_interfaces.json", {"status": "observed", "workers": backend.rpc("benchmark_inspect_runtime"),
                                                    "offline_engine": backend.metadata})
        for args in args_list:
            run_dataset(args)
    reports = {name: json.loads((root / name / "acceptance_report.json").read_text()) for name in names}
    dump_json(root / "acceptance_report.json", reports)
    print("Offline benchmark completed: " + str(root), flush=True)


if __name__ == "__main__":
    main()
