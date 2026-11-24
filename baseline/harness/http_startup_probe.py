"""Opt-in HTTP startup tracing; no model execution or request content collection."""
from __future__ import annotations

import asyncio
import faulthandler
import functools
import inspect
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

MODULES = {"uvicorn.server", "vllm.entrypoints.launchers.launcher", "vllm.entrypoints.launcher"}
_stack_file = None


def emit(event, **fields):
    directory = os.environ.get("VLLM_BENCHMARK_STARTUP_DIR")
    if not directory:
        return
    try:
        row = dict(event=event, pid=os.getpid(), wall_time_s=time.time(), **fields)
        with (Path(directory) / f"http_startup_{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:
        print("HTTP STARTUP EVIDENCE WRITE FAILED: " + type(exc).__name__, file=sys.stderr)


def socket_state(sock):
    try:
        return {"fd": sock.fileno(), "address": sock.getsockname(),
                "accepting": bool(sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))}
    except Exception as exc:
        return {"error_type": type(exc).__name__, "detail": str(exc)}


def state(owner, arguments):
    sockets = arguments.get("sockets")
    if sockets is None:
        sock = arguments.get("sock")
        sockets = [sock] if sock is not None else []
    listeners = []
    for server in getattr(owner, "servers", []) or []:
        listeners.extend(socket_state(sock) for sock in (getattr(server, "sockets", []) or []))
    config = getattr(owner, "config", None)
    try:
        namespace = os.readlink("/proc/self/ns/net")
    except OSError:
        namespace = None
    return {"passed_sockets": [socket_state(sock) for sock in sockets], "listeners": listeners,
            "started": getattr(owner, "started", None), "should_exit": getattr(owner, "should_exit", None),
            "peer_loads_enabled": getattr(owner, "peer_loads", None) is not None,
            "host": getattr(config, "host", None), "port": getattr(config, "port", None),
            "net_namespace": namespace, "loop": type(asyncio.get_running_loop()).__name__}


def _dump_tasks():
    try:
        for task in asyncio.all_tasks():
            frames = task.get_stack(limit=12)
            emit("async_task_stack", coroutine=getattr(task.get_coro(), "__qualname__", "unknown"),
                 stack=["".join(traceback.format_stack(frame, limit=8)) for frame in frames])
    except Exception as exc:
        emit("async_stack_error", error_type=type(exc).__name__)


def arm_stacks():
    global _stack_file
    if _stack_file is not None:
        return
    try:
        directory = Path(os.environ["VLLM_BENCHMARK_STARTUP_DIR"])
        _stack_file = (directory / f"thread_stacks_{os.getpid()}.txt").open("a", encoding="utf-8")
        faulthandler.dump_traceback_later(20, file=_stack_file, repeat=False)
        asyncio.get_running_loop().call_later(20, _dump_tasks)
        emit("stack_capture_armed", after_s=20)
    except Exception as exc:
        emit("stack_capture_error", error_type=type(exc).__name__)


def wrap_async(owner, name):
    original = owner.__dict__.get(name)
    if not inspect.iscoroutinefunction(original) or getattr(original, "_startup_probe", False):
        return
    signature = inspect.signature(original)
    label = original.__module__ + "." + original.__qualname__
    @functools.wraps(original)
    async def wrapped(*args, **kwargs):
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
            server = bound.get("self")
            arm_stacks()
            emit("enter", function=label, state=state(server, bound))
        except Exception as exc:
            bound, server = {}, None
            emit("probe_error", function=label, error_type=type(exc).__name__)
        try:
            result = await original(*args, **kwargs)
        except BaseException as exc:
            emit("exception", function=label, error_type=type(exc).__name__, traceback=traceback.format_exc(limit=20))
            raise
        else:
            try:
                emit("return", function=label, state=state(server, bound))
            except Exception as exc:
                emit("probe_error", function=label, error_type=type(exc).__name__)
            return result
    wrapped._startup_probe = True
    setattr(owner, name, wrapped)
    emit("hook_installed", function=label, signature=str(signature))


def instrument(module):
    if not os.environ.get("VLLM_BENCHMARK_STARTUP_DIR"):
        return
    for value in list(vars(module).values()):
        if inspect.isclass(value) and value.__module__ == module.__name__ and value.__name__ in ("Server", "NoSignalServer"):
            for name in ("serve", "_serve", "startup"):
                wrap_async(value, name)
    wrap_async(module, "serve_http")
