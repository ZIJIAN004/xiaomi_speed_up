#!/usr/bin/env python3
"""Verify actual hook evidence from the short runtime check; never certify baseline quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    run = root / "omnidocbench"
    errors = []
    def read(path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path), "error_type": type(exc).__name__})
            return default
    def jsonl(path):
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path), "error_type": type(exc).__name__})
            return []
    runtime = read(root / "runtime_interfaces.json", {})
    records = jsonl(run / "per_sample_metrics.jsonl")
    calls = jsonl(run / "server_processor_calls.jsonl")
    processor_status = read(run / "server_processor_status.json", {})
    telemetry = read(run / "worker_telemetry.json", {})
    ranges = read(run / "profile_ranges.json", {})
    profiles = read(run / "profile_metrics.json", [])
    workers = telemetry.get("results", [])
    checks = []
    def check(name, passed, evidence):
        checks.append({"check": name, "status": "verified" if passed else "not_verified", "evidence": evidence})
    check("loaded_runner_interfaces", bool(runtime.get("workers")) and all(
        bool(w.get("runner", {}).get("methods", {}).get("execute_model", {}).get("signature")) for w in runtime.get("workers", [])), runtime)
    check("eight_request_smoke_completed", len(records) == 8 and all(r.get("success") for r in records),
          {"records": len(records), "successes": sum(bool(r.get("success")) for r in records)})
    for kind in ("multimodal_apply", "hf_processor"):
        selected = [r for r in calls if r.get("kind") == kind]
        check(kind + "_actually_called", bool(selected) and all(r.get("success") for r in selected),
              {"calls": len(selected), "classes": sorted({r.get("processor_class", "unknown") for r in selected})})
    processor_calls = [r for r in calls if r.get("kind") in ("multimodal_apply", "hf_processor")]
    check("request_context_encloses_processor", bool(processor_calls) and all(r.get("request_id") for r in processor_calls),
          {"processor_calls": len(processor_calls), "with_request_id": sum(bool(r.get("request_id")) for r in processor_calls),
           "note": "not_verified means process_inputs does not enclose all preprocessing; use source call chain for another entry"})
    check("processor_hook_setup_without_errors", bool(processor_status.get("hook_metadata")) and not
          any(r.get("event") in ("hook_error", "unsupported") for r in processor_status.get("hook_metadata", []))
          and not processor_status.get("parse_errors"),
          {"hook_metadata": processor_status.get("hook_metadata"), "parse_errors": processor_status.get("parse_errors")})
    schemas = [w.get("steps", {}).get("observed_schemas", {}) for w in workers]
    check("actual_scheduler_attributes", bool(schemas) and all(
        all(s.get("scheduler_output", {}).get("required_attributes", {}).get(name) is True for name in
            ("scheduled_new_reqs", "scheduled_cached_reqs", "num_scheduled_tokens", "total_num_scheduled_tokens"))
        and all(s.get("new_request", {}).get(name) is True for name in
            ("req_id", "prompt_token_ids", "num_computed_tokens", "mm_features")) for s in schemas), schemas)
    check("input_ids_correlated_to_usage", len(records) == 8 and all(r.get("worker_input_length_verified") for r in records),
          {"verified": sum(bool(r.get("worker_input_length_verified")) for r in records), "total": len(records)})
    check("output_ids_materialized_and_match_usage", bool(records) and all(
        r.get("generated_token_classification_source") in ("actual_worker_sampled_ids_matched_to_server_usage", "actual_offline_delta_token_ids") for r in records),
        {"output_types": [w.get("steps", {}).get("output_type_step_counts") for w in workers],
         "classified_requests": sum(r.get("generated_visible_tokens") is not None for r in records),
         "note": "async output absence is not synchronized away; missing evidence needs an adapter"})
    installed = [r.get("installed", {}) for r in ranges.get("results", [])]
    for stage in ("vision_encoder", "projector", "sampling"):
        check(stage + "_hook_installed", bool(installed) and all(stage in item for item in installed), installed)
    for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling"):
        observed = [p.get(stage + "_time_ms") for p in profiles if p.get("success") and not p.get("profile_stop_error")]
        check(stage + "_cuda_events_observed", any(isinstance(v, (int, float)) and v > 0 for v in observed), observed)
    check("real_cuda_trace_exported", any(t.get("kernel_count", 0) for p in profiles for t in p.get("trace_analysis", [])),
          [{"sample_id": p.get("sample_id"), "files": p.get("trace_files"), "profile_error": p.get("profile_error"),
            "stop_error": p.get("profile_stop_error")} for p in profiles])
    check("cuda_operator_attribution", any(t.get("cuda_operator_attributed_kernel_count", 0) > 0
          for p in profiles for t in p.get("trace_analysis", [])),
          [t.get("cuda_operator_attributed_kernel_count") for p in profiles for t in p.get("trace_analysis", [])])
    return {"diagnostic_only": True, "all_runtime_checks_verified": all(c["status"] == "verified" for c in checks),
            "checks": checks, "read_errors": errors,
            "limits": "8 copies of one image, output capped at 32; not quality, throughput or full-length baseline acceptance. Request-context propagation is checked only on real observed calls."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    for check in result["checks"]:
        print(check["status"] + ": " + check["check"])
    print("Saved:", args.output)
    raise SystemExit(0 if result["all_runtime_checks_verified"] else 1)


if __name__ == "__main__":
    main()
