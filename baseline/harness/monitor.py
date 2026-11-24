"""Fixed-cadence sampling with compact RAM scalars and bounded full snapshots."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from baseline.harness.monitor_storage import ScalarStore


def get_url(url, timeout=10):
    if url.startswith("offline://single/"):
        from baseline.harness.offline_backend import active
        return active().get(url.split("offline://single/", 1)[1])
    # Local inference traffic must not be sent to an environment HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def rpc(base_url, method, args=(), timeout=60):
    if base_url == "offline://single":
        from baseline.harness.offline_backend import active
        return active().rpc(method, args, timeout)
    payload = json.dumps({"method": method, "args": list(args)}).encode()
    request = urllib.request.Request(base_url + "/collective_rpc", data=payload,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response).get("results", [])


def process_snapshot(server_pid, previous=None):
    """Linux server process tree only, not unrelated jobs sharing the GPU."""
    now = time.monotonic()
    ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    entries = {}
    for folder in Path("/proc").glob("[0-9]*"):
        try:
            tail = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(folder.name)
            status = (folder / "status").read_text()
            entries[pid] = {"ppid": int(tail[1]), "start_ticks": int(tail[19]),
                            "cpu_seconds": (int(tail[11]) + int(tail[12])) / ticks,
                            "rss_bytes": int(next((s.split()[1] for s in status.splitlines()
                                                   if s.startswith("VmRSS:")), "0")) * 1024}
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    selected = {server_pid} if server_pid else set()
    while True:
        children = {pid for pid, info in entries.items() if info["ppid"] in selected}
        if children <= selected:
            break
        selected |= children
    chosen = {pid: info for pid, info in entries.items() if pid in selected}
    cpu = None
    if previous and now > previous[0]:
        # Only compare surviving PIDs; PID churn is not an inferred negative CPU delta.
        cpu = sum(max(0, info["cpu_seconds"] - previous[1][pid]["cpu_seconds"])
                  for pid, info in chosen.items() if pid in previous[1] and info["start_ticks"] == previous[1][pid]["start_ticks"]) / (now - previous[0]) * 100
    for pid, info in chosen.items():
        old = previous[1].get(pid) if previous else None
        info["cpu_percent"] = (max(0, info["cpu_seconds"] - old["cpu_seconds"]) / (now - previous[0]) * 100
                               if old and old["start_ticks"] == info["start_ticks"] and now > previous[0] else None)
    system = None
    system_ticks = None
    try:
        system_ticks = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        if previous and previous[2]:
            delta = [a - b for a, b in zip(system_ticks, previous[2])]
            system = (1 - (delta[3] + delta[4]) / sum(delta)) * 100 if sum(delta) else None
    except OSError:
        pass
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapFree", "SwapTotal"):
                mem[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    record = {"server_process_cpu_percent": cpu, "system_cpu_percent": system,
              "server_process_rss_bytes": sum(v["rss_bytes"] for v in chosen.values()) if chosen else None,
              "server_pids": sorted(chosen), "server_process_samples": chosen,
              "rss_scope": "sum of process RSS, shared pages can be counted more than once; not PSS",
              "cpu_percent_scope": "per logical core 100 percent; process tree can exceed 100 percent",
              "system_memory": mem,
              "pinned_memory_bytes": None, "windows_host_memory_bytes": None}
    return record, (now, chosen, system_ticks)


class Monitor:
    def __init__(self, output, base_url, gpu, server_pid=None, interval=1.0, dcgm_url=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.base_url, self.gpu, self.server_pid, self.interval = base_url, gpu, server_pid, interval
        if interval <= 0 or not math.isfinite(interval):
            raise ValueError("monitor interval must be finite and positive")
        self.stop_event = threading.Event()
        self.phase_lock = threading.Lock()
        self._phase = "initialization"
        self.phase_epoch = 0
        self.store = ScalarStore()
        self.snapshot_phases = set()
        self.last_checkpoint = time.monotonic()
        self.last_full = None
        self.previous = None
        self.client_previous = None
        self.dcgm_url = dcgm_url
        self.windows_powershell = shutil.which("powershell.exe") if os.environ.get("WSL_DISTRO_NAME") else None
        self.windows_last_sample_s = float("-inf")
        self.thread = None
        self.errors = []

    @property
    def phase(self):
        with self.phase_lock:
            return self._phase

    @phase.setter
    def phase(self, value):
        with self.phase_lock:
            self._phase = value
            self.phase_epoch += 1

    def sample(self, index):
        from baseline.harness.collect_results import read_prometheus
        start = time.monotonic()
        with self.phase_lock:
            phase, epoch = self._phase, self.phase_epoch
        raw, raw_dcgm = None, None
        record, self.previous = process_snapshot(self.server_pid, self.previous)
        client, self.client_previous = process_snapshot(os.getpid(), self.client_previous)
        record["client_process_cpu_percent"] = client["server_process_cpu_percent"]
        record["client_process_rss_bytes"] = client["server_process_rss_bytes"]
        record.update(monotonic_s=start, wall_time_s=time.time(), phase=phase, gpu_scope="whole_device_shared_if_other_jobs")
        if self.windows_powershell and start - self.windows_last_sample_s >= 10:
            self.windows_last_sample_s = start
            try:
                command = "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"
                host = subprocess.run([self.windows_powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                                      capture_output=True, text=True, check=True, timeout=5)
                memory = json.loads(host.stdout.lstrip("﻿"))
                total, free = int(memory["TotalVisibleMemorySize"]) * 1024, int(memory["FreePhysicalMemory"]) * 1024
                record.update(windows_host_memory_bytes=total - free, windows_host_total_memory_bytes=total,
                              windows_host_free_memory_bytes=free, windows_host_observation_wall_s=time.time())
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                record["windows_host_memory_error"] = str(exc)
        query = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem"
        try:
            result = subprocess.run(["nvidia-smi", f"--id={self.gpu}", f"--query-gpu={query}",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            values = next(csv.reader(io.StringIO(result.stdout)))
            names = ["gpu_utilization_percent", "gpu_memory_utilization_percent", "gpu_memory_used_mib",
                     "gpu_memory_total_mib", "gpu_power_w", "gpu_temperature_c", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz"]
            for key, value in zip(names, values):
                try:
                    record[key] = float(value.strip())
                except ValueError:
                    record[key] = None
        except (OSError, subprocess.SubprocessError, StopIteration) as exc:
            record["gpu_sampling_error"] = str(exc)
        record.update(sm_utilization_percent=None, tensor_core_utilization_percent=None,
                      gpu_active_time_ms=None, gpu_idle_time_ms=None)
        if self.dcgm_url:
            try:
                raw_dcgm = get_url(self.dcgm_url, timeout=5)
                for line in raw_dcgm.splitlines():
                    match = re.match(r'(DCGM_FI_PROF_SM_ACTIVE|DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)\{(.*?)\}\s+(\S+)', line)
                    if match and re.search(r'(?:^|,)\s*gpu="' + re.escape(str(self.gpu)) + r'"(?:,|$)', match[2]):
                        value = float(match[3])
                        if 0 <= value <= 1:
                            key = "sm_utilization_percent" if match[1] == "DCGM_FI_PROF_SM_ACTIVE" else "tensor_core_utilization_percent"
                            record[key] = value * 100
            except Exception as exc:
                record["dcgm_error"] = str(exc)
        try:
            raw = get_url(self.base_url + "/metrics", timeout=5)
        except Exception as exc:
            record["metrics_error"] = str(exc)
        record["sample_end_monotonic_s"] = time.monotonic()
        with self.phase_lock:
            record["phase_end"] = self._phase if epoch == self.phase_epoch else "transition"
        record["sampling_duration_s"] = record["sample_end_monotonic_s"] - start
        record["metrics_sampling_success"] = int(raw is not None)
        record["gpu_sampling_success"] = int("gpu_sampling_error" not in record)
        record["dcgm_sampling_success"] = int(raw_dcgm is not None) if self.dcgm_url else None
        record["windows_sampling_success"] = (int("windows_host_memory_error" not in record)
                                               if "windows_host_memory_error" in record or
                                               "windows_host_observation_wall_s" in record else None)
        self.store.append(record, read_prometheus(raw) if raw is not None else {},
                          read_prometheus(raw_dcgm) if raw_dcgm is not None else {})
        self.last_full = (record, raw, raw_dcgm)
        if phase not in self.snapshot_phases:
            self._snapshot("phase_" + str(len(self.snapshot_phases) + 1), self.last_full)
            self.snapshot_phases.add(phase)
        if self.store.count == 1 or time.monotonic() - self.last_checkpoint >= 60:
            self.store.checkpoint(self.output / "monitor_summary.json", self.interval, self.errors)
            self.last_checkpoint = time.monotonic()

    def _snapshot(self, name, full):
        record, raw, dcgm = full
        from baseline.harness.metrics import dump_json
        dump_json(self.output / (name + ".json"), record)
        if raw is not None:
            (self.output / (name + ".prom")).write_text(raw, encoding="utf-8")
        if dcgm is not None:
            (self.output / (name + "_dcgm.prom")).write_text(dcgm, encoding="utf-8")

    def _run(self):
        # Absolute deadlines: collection time does not add to the nominal period.
        # Slow rounds skip missed slots instead of generating catch-up bursts.
        deadline = time.monotonic()
        index = 0
        while not self.stop_event.is_set():
            try:
                self.sample(index)
            except Exception as exc:
                if len(self.errors) < 100:
                    self.errors.append(str(exc))
            index += 1
            deadline += self.interval
            now = time.monotonic()
            if deadline < now:
                deadline += (math.floor((now - deadline) / self.interval) + 1) * self.interval
            self.stop_event.wait(max(0., deadline - time.monotonic()))
        # No per-round disk records; preserve all scalar values once at shutdown.
        try:
            if self.last_full is not None:
                self._snapshot("final", self.last_full)
            self.store.save(self.output / "scalars.zip")
            self.store.checkpoint(self.output / "monitor_summary.json", self.interval, self.errors, complete=True)
        except Exception as exc:
            self.errors.append("monitor finalization failed: " + str(exc))

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
