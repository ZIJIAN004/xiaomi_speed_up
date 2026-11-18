"""Opt-in import-time instrumentation of real vLLM multimodal processor calls.

No model inputs, images, prompt text, output text or token IDs are logged.
"""
from __future__ import annotations

import contextvars
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

_active = contextvars.ContextVar("benchmark_processor_active", default={})
_lock = threading.Lock()
_warned = False


def _after_fork():
    global _lock
    _lock = threading.Lock()
    _active.set({})


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def emit(record):
    global _warned
    directory = os.environ.get("VLLM_BENCHMARK_PROCESSOR_DIR")
    if not directory:
        return
    try:
        data = (json.dumps(dict(record, pid=os.getpid()), allow_nan=False) + "\n").encode()
        with _lock:
            path = Path(directory) / ("processor_" + str(os.getpid()) + ".jsonl")
            with path.open("ab") as stream:
                stream.write(data)
    except Exception as exc:
        if not _warned:
            print("BENCHMARK PROCESSOR LOG UNAVAILABLE: " + type(exc).__name__, file=sys.stderr)
            _warned = True


def wrap_method(cls, name, kind):
    original = cls.__dict__.get(name)
    if original is None or getattr(original, "_benchmark_processor_wrapped", False):
        return
    if not inspect.isfunction(original) or inspect.iscoroutinefunction(original):
        emit(dict(event="unsupported", processor_class=cls.__qualname__, method=name))
        return

    @functools.wraps(original)
    def measured(self, *args, **kwargs):
        active = _active.get()
        # Subclass -> super() is one invocation, not two independent costs.
        if kind in active:
            return original(self, *args, **kwargs)
        call_id = uuid.uuid4().hex
        token = _active.set(dict(active, **{kind: call_id}))
        cpu_start = time.thread_time()
        start = time.perf_counter()
        success, error_type = False, None
        try:
            value = original(self, *args, **kwargs)
            success = True
            return value
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            end = time.perf_counter()
            cpu_end = time.thread_time()
            _active.reset(token)
            emit(dict(event="call", kind=kind, call_id=call_id,
                      parent_apply_call_id=active.get("multimodal_apply"),
                      processor_class=type(self).__module__ + "." + type(self).__qualname__,
                      method=name, start_s=start, end_s=end,
                      wall_time_ms=(end - start) * 1000,
                      thread_cpu_time_ms=(cpu_end - cpu_start) * 1000,
                      success=success, error_type=error_type,
                      request_id=None, request_id_reason="processor API has no verified request-ID context"))
    measured._benchmark_processor_wrapped = True
    setattr(cls, name, measured)
    emit(dict(event="hook_installed", module=cls.__module__, processor_class=cls.__qualname__,
              method=name, kind=kind, signature=str(inspect.signature(original))))


def instrument_module(module):
    for value in list(vars(module).values()):
        if not inspect.isclass(value) or value.__module__ != module.__name__:
            continue
        if not any(base.__name__ == "BaseMultiModalProcessor" for base in value.__mro__):
            continue
        wrap_method(value, "apply", "multimodal_apply")
        wrap_method(value, "_call_hf_processor", "hf_processor")


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            instrument_module(module)
        except Exception as exc:
            emit(dict(event="hook_error", module=module.__name__, error_type=type(exc).__name__))

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == "vllm.multimodal.processing" or
                fullname.startswith("vllm.multimodal.processing.") or
                fullname.startswith("vllm.model_executor.models.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
        for name, module in list(sys.modules.items()):
            if module and (name.startswith("vllm.multimodal.processing") or name.startswith("vllm.model_executor.models.")):
                instrument_module(module)
        emit(dict(event="bootstrap", clock="time.perf_counter", scope="actual server process; no CUDA synchronization"))
