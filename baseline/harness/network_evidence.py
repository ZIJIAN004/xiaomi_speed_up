"""Read-only Linux socket evidence restricted to the launched process tree."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def process_tree(root_pid, proc=Path("/proc")):
    parents = {}
    for path in proc.glob("[0-9]*/stat"):
        try:
            tail = path.read_text().rsplit(")", 1)[1].split()
            parents[int(path.parent.name)] = int(tail[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    while True:
        children = {pid for pid, parent in parents.items() if parent in selected}
        if children <= selected:
            return sorted(selected)
        selected |= children


def tcp_entries(path):
    result = []
    try:
        for line in path.read_text().splitlines()[1:]:
            parts = line.split()
            result.append({"local_hex": parts[1].split(":")[0], "port": int(parts[1].split(":")[1], 16),
                           "state_hex": parts[3], "listening": parts[3] == "0A", "inode": parts[9]})
    except (OSError, ValueError, IndexError) as exc:
        return {"error_type": type(exc).__name__}
    return {"entries": result}


def snapshot(root_pid, port, proc=Path("/proc"), include_ss=True):
    report = {"wall_time_s": time.time(), "target_port": port, "root_pid": root_pid, "processes": []}
    try:
        report["client_net_namespace"] = os.readlink(proc / "self/ns/net")
    except OSError as exc:
        report["client_net_namespace_error"] = type(exc).__name__
    report["client_namespace_target_entries"] = []
    for name in ("tcp", "tcp6"):
        table = tcp_entries(proc / "self/net" / name)
        report["client_namespace_target_entries"].extend(
            dict(row, family=name) for row in table.get("entries", []) if row["port"] == port)
    for pid in process_tree(root_pid, proc):
        folder = proc / str(pid)
        item = {"pid": pid, "socket_fds": {}, "owned_tcp_entries": []}
        for name, target in (("net_namespace", folder / "ns/net"), ("executable", folder / "exe")):
            try:
                item[name] = os.readlink(target)
            except OSError as exc:
                item[name + "_error"] = type(exc).__name__
        try:
            item["name"] = (folder / "comm").read_text().strip()
        except OSError:
            pass
        try:
            for fd in (folder / "fd").iterdir():
                try:
                    link = os.readlink(fd)
                    if link.startswith("socket:["):
                        item["socket_fds"][fd.name] = link[8:-1]
                except OSError:
                    continue
        except OSError as exc:
            item["fd_error"] = type(exc).__name__
        owned = set(item["socket_fds"].values())
        for name in ("tcp", "tcp6"):
            table = tcp_entries(folder / "net" / name)
            item["owned_tcp_entries"].extend(dict(row, family=name) for row in table.get("entries", []) if row["inode"] in owned)
        report["processes"].append(item)
    if include_ss:
        try:
            result = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=3, check=False)
            report["ss"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["ss"] = {"error_type": type(exc).__name__}
    return report
