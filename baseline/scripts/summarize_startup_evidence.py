#!/usr/bin/env python3
"""Collect startup evidence without claiming a root cause from missing observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root):
    root = Path(root)
    evidence = root / "startup_evidence"
    errors = []
    def read(path, default):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path), "error_type": type(exc).__name__})
            return default
    events = []
    for path in sorted(evidence.glob("http_startup_*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                events.append(json.loads(line))
            except ValueError:
                errors.append({"file": str(path), "line": number, "error_type": "invalid_json"})
    events.sort(key=lambda item: item.get("wall_time_s", 0))
    network = read(evidence / "network_final.json", {})
    readiness = read(root / "server_startup.json", {})
    exceptions = [r for r in events if r.get("event") == "exception" and r.get("error_type") != "CancelledError"]
    lifecycle = [r for r in events if r.get("event") in ("enter", "return", "exception", "probe_error")]
    return {"diagnostic_only": True, "readiness": readiness,
            "http_hooks_observed": any(r.get("event") == "hook_installed" for r in events),
            "http_lifecycle": lifecycle, "http_task_exceptions": exceptions,
            "network_final": network,
            "different_network_namespace_pids": [p.get("pid") for p in network.get("processes", [])
                if p.get("net_namespace") and network.get("client_net_namespace") and p["net_namespace"] != network["client_net_namespace"]],
            "client_namespace_target_listeners": [r for r in network.get("client_namespace_target_entries", []) if r.get("listening")],
            "async_task_stacks": [r for r in events if r.get("event") == "async_task_stack"],
            "thread_stack_files": [str(path) for path in sorted(evidence.glob("thread_stacks_*.txt"))],
            "read_errors": errors,
            "scope": "network_final captured before cleanup; lifecycle can include later shutdown. No inference requests. Missing evidence is not proof of cause."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.run_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("Startup evidence report:", args.output)
    print("Readiness:", report["readiness"].get("status", "unavailable"))
    print("HTTP hooks observed:", report["http_hooks_observed"])
    print("HTTP task exceptions:", len(report["http_task_exceptions"]))
    print("Target-port listeners before cleanup:", len(report["client_namespace_target_listeners"]))


if __name__ == "__main__":
    main()
