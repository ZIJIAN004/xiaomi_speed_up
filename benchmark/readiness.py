"""Explicit local HTTP readiness diagnostics, separate from model startup logs."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def probe(opener, url, timeout):
    started = time.perf_counter()
    try:
        with opener.open(url, timeout=timeout) as response:
            result = {"status": response.status, "error": None}
    except urllib.error.HTTPError as exc:
        result = {"status": exc.code, "error": "HTTPError"}
        exc.close()
    except (OSError, urllib.error.URLError) as exc:
        result = {"status": None, "error": type(exc).__name__, "detail": str(getattr(exc, "reason", exc))}
    return dict(result, elapsed_s=time.perf_counter() - started)


def wait_for_server(base_url, pid, launched_at, output, timeout=1200, interval=5,
                    opener=None, clock=time.monotonic, sleep=time.sleep, alive=None,
                    diagnostic_dir=None, server_log=None, application_ready_grace=0):
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or parsed.username:
        raise ValueError("readiness probes must target a local HTTP server")
    if timeout <= 0 or interval <= 0:
        raise ValueError("readiness timeout and interval must be positive")
    output = Path(output)
    checks = output.with_name("readiness_checks.jsonl")
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    alive = alive or (lambda: os.kill(pid, 0))
    deadline = clock() + timeout
    diagnostic_dir = Path(diagnostic_dir) if diagnostic_dir else None
    if diagnostic_dir:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
    def capture(label):
        if not diagnostic_dir:
            return
        try:
            from benchmark.network_evidence import snapshot
            data = snapshot(pid, parsed.port or 80)
            data["target_url"] = base_url
            (diagnostic_dir / ("network_" + label + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print("Startup evidence capture failed: " + repr(exc), flush=True)
    last_capture = float("-inf")
    app_ready_at = None
    last = {}
    attempt = 0
    failure = "readiness deadline exceeded"
    print(f"Waiting for local vLLM HTTP readiness, deadline={timeout}s; diagnostics: {checks}", flush=True)
    with checks.open("x", encoding="utf-8") as stream:
        while clock() < deadline:
            attempt += 1
            try:
                alive()
            except ProcessLookupError:
                failure = "launcher-recorded vLLM PID exited during startup"
                break
            if diagnostic_dir and clock() - last_capture >= 20:
                capture(str(attempt))
                last_capture = clock()
            health = probe(opener, base_url + "/health", min(5, max(.01, deadline - clock())))
            last = {"attempt": attempt, "wall_time_s": time.time(), "target_url": base_url, "health": health}
            if health["status"] == 200 and clock() < deadline:
                last["models"] = probe(opener, base_url + "/v1/models", min(5, max(.01, deadline - clock())))
            stream.write(json.dumps(last, allow_nan=False) + "\n")
            stream.flush()
            print("readiness: " + json.dumps(last, ensure_ascii=False), flush=True)
            if health["status"] == 200 and last.get("models", {}).get("status") == 200:
                ready = time.time()
                data = {"status": "ready", "server_start_wall_s": launched_at, "server_ready_wall_s": ready,
                        "cold_start_latency_s": ready - launched_at, "server_pid": pid,
                        "cold_start_scope": "launch through /health and /v1/models HTTP 200; includes load/compile/graph capture",
                        "model_load_time_s": None, "cuda_initialization_time_s": None, "last_probe": last}
                capture("final")
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return data
            codes = [health["status"], last.get("models", {}).get("status")]
            if any(code in (401, 403) for code in codes):
                failure = "HTTP authentication/access denied; inspect API key or access policy, not model loading"
                break
            if health["status"] == 200 and last.get("models", {}).get("status") == 404:
                failure = "/health ready but /v1/models unavailable (HTTP 404); inspect route configuration"
                break
            if application_ready_grace > 0 and server_log:
                try:
                    app_ready = "Application startup complete." in Path(server_log).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    app_ready = False
                if app_ready and app_ready_at is None:
                    app_ready_at = clock()
                    print(f"Application startup complete detected; allowing {application_ready_grace}s for HTTP readiness", flush=True)
                if app_ready_at is not None and clock() - app_ready_at >= application_ready_grace:
                    failure = "application startup completed but HTTP readiness still failed after diagnostic grace"
                    break
            remaining = deadline - clock()
            if remaining > 0:
                sleep(min(interval, remaining))
    capture("final")
    data = {"status": "failed", "reason": failure, "last_probe": last, "server_pid": pid,
            "server_start_wall_s": launched_at, "cold_start_latency_s": None,
            "diagnostics": str(checks), "proxy_policy": "environment HTTP proxies disabled"}
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    raise RuntimeError(f"vLLM readiness failed: {failure}. Last probe: {json.dumps(last)}. See {checks} and logs/vllm_server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--launched-at", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--application-ready-grace", type=float, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name("readiness_checks.jsonl").exists():
        parser.error("readiness output files must be new")
    try:
        wait_for_server(args.base_url, args.pid, args.launched_at, args.output, args.timeout,
                        diagnostic_dir=args.diagnostic_dir, server_log=args.server_log,
                        application_ready_grace=args.application_ready_grace)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
