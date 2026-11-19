"""Opt-in vLLM worker extension. RPC is localhost-only and disabled unless explicitly enabled.

No synchronize or CUDA Events are inserted into steady-state token execution.
Memory peaks are run-wide at concurrency>1, not falsely attributed to individual requests.
"""
from __future__ import annotations

import functools
import importlib
import json
import os
import threading
import time
from contextlib import contextmanager


class BenchmarkWorkerExtension:
    @contextmanager
    def _benchmark_range(self, stage, annotation=None):
        import torch
        event = None
        if getattr(self, "_benchmark_events_enabled", False):
            stream = torch.cuda.current_stream(self.device)
            if not torch.cuda.is_current_stream_capturing():
                key = int(stream.cuda_stream)
                if key not in self._benchmark_event_anchors:
                    anchor = torch.cuda.Event(enable_timing=True)
                    anchor.record(stream)
                    self._benchmark_event_anchors[key] = anchor
                tid = threading.get_ident()
                stack = self._benchmark_event_stacks.setdefault(tid, [])
                event = {"id": len(self._benchmark_events), "stage": stage, "stream": key,
                         "parent_id": stack[-1] if stack else None, "thread_id": tid,
                         "start": torch.cuda.Event(enable_timing=True), "end": torch.cuda.Event(enable_timing=True),
                         "host_start_ns": time.perf_counter_ns(), "host_end_ns": None,
                         "final_prefill_sampling": stage == "sampling" and getattr(self, "_benchmark_step_finishes_prompt", False)}
                self._benchmark_events.append(event)
                event["start"].record(stream)
                stack.append(event["id"])
            else:
                self._benchmark_event_skipped_capture += 1
        try:
            with torch.profiler.record_function("benchmark::" + (annotation or stage)):
                yield
        finally:
            if event is not None:
                event["host_end_ns"] = time.perf_counter_ns()
                self._benchmark_event_stacks[event["thread_id"]].pop()
                try:
                    event["end"].record(stream)
                except RuntimeError as exc:
                    event["error"] = str(exc)

    def benchmark_event_timing(self, enable="true"):
        """Representative requests only. One final synchronize after profiler stop, never per token."""
        import torch
        if enable == "true":
            self._benchmark_events_enabled = True
            self._benchmark_events = []
            self._benchmark_event_anchors = {}
            self._benchmark_event_stacks = {}
            self._benchmark_event_skipped_capture = 0
            self._benchmark_step_finishes_prompt = False
            return {"enabled": True, "scope": "representative_only_current_stream_cuda_events"}
        self._benchmark_events_enabled = False
        torch.cuda.synchronize(self.device)
        events = []
        for event in getattr(self, "_benchmark_events", []):
            if event.get("error"):
                continue
            anchor = self._benchmark_event_anchors[event["stream"]]
            events.append({k: v for k, v in event.items() if k not in ("start", "end")})
            events[-1].update(start_ms=anchor.elapsed_time(event["start"]),
                              end_ms=anchor.elapsed_time(event["end"]),
                              host_wall_time_ms=((event.get("host_end_ns") - event.get("host_start_ns")) / 1e6
                                                 if event.get("host_end_ns") is not None else None))
        self._benchmark_events = []
        self._benchmark_event_anchors = {}
        return {"events": events, "scope": "per-stream CUDA elapsed timeline, includes stream idle gaps; not kernel work sum",
                "skipped_graph_capture_ranges": getattr(self, "_benchmark_event_skipped_capture", 0),
                "measurement_sync": "one final synchronize; caller invokes after stop_profile and records whether stop succeeded"}

    def benchmark_profile_limit(self, max_steps="64", trace_dir="", restore="false"):
        """Bound trace memory without truncating the generated output."""
        import torch
        if restore == "true":
            self._benchmark_legacy_profiler_step = False
            original = getattr(self, "_benchmark_original_profiler", None)
            exported = None
            profiler = getattr(self, "profiler", None)
            if original is not None:
                try:
                    if not getattr(self, "_benchmark_trace_exported", False):
                        import time
                        from pathlib import Path
                        path = Path(trace_dir) / f"benchmark_{os.getpid()}_{time.time_ns()}.pt.trace.json"
                        profiler.export_chrome_trace(str(path))
                        exported = str(path)
                finally:
                    self.profiler = original
                    self._benchmark_original_profiler = None
            return {"restored": True, "fallback_trace": exported}
        steps = int(max_steps)
        if steps < 1:
            raise ValueError("profile max steps must be positive")
        config = getattr(getattr(self, "vllm_config", None), "profiler_config", None)
        configured = getattr(config, "max_iterations", 0)
        if configured and configured > 0:
            self._benchmark_legacy_profiler_step = False
            return {"bounded": True, "max_steps": configured, "source": "installed_vllm_profiler_config"}
        profiler = getattr(self, "profiler", None)
        profile_class = getattr(torch.profiler, "profile", None)
        if not isinstance(profile_class, type) or not isinstance(profiler, profile_class):
            return {"bounded": False, "reason": "installed profiler has no supported bounded interface; need version-specific adapter"}
        self._benchmark_original_profiler = profiler
        self._benchmark_trace_exported = False
        from pathlib import Path
        import time
        import shutil
        trace_path = Path(trace_dir) / f"benchmark_{os.getpid()}_{time.time_ns()}.pt.trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        def save(profile):
            profile.export_chrome_trace(str(trace_path))
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=True, with_flops=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=steps, repeat=1), on_trace_ready=save)
        # Kineto permits one export. If vLLM requests another name on stop,
        # relocate/re-encode the canonical payload rather than keeping copies.
        export_original = self.profiler.export_chrome_trace
        def export_once(path):
            nonlocal trace_path
            if not self._benchmark_trace_exported:
                export_original(str(trace_path))
                self._benchmark_trace_exported = True
            requested = Path(path)
            if requested.resolve() != trace_path.resolve():
                requested.parent.mkdir(parents=True, exist_ok=True)
                if requested.exists():
                    raise FileExistsError("refusing to overwrite another profiler artifact: " + str(requested))
                source_gz = str(trace_path).endswith(".gz")
                target_gz = str(requested).endswith(".gz")
                if source_gz != target_gz:
                    import gzip
                    # Keep the source until conversion succeeds; only our newly
                    # generated source is removed, never historical run data.
                    with requested.open("xb") as raw_target:
                        try:
                            opener = gzip.open if source_gz else open
                            with opener(trace_path, "rb") as source:
                                if target_gz:
                                    with gzip.GzipFile(fileobj=raw_target, mode="wb") as target:
                                        shutil.copyfileobj(source, target)
                                else:
                                    shutil.copyfileobj(source, raw_target)
                        except BaseException:
                            raw_target.close()
                            requested.unlink()
                            raise
                    trace_path.unlink()
                else:
                    trace_path.rename(requested)
                trace_path = requested
            return str(trace_path)
        self.profiler.export_chrome_trace = export_once
        self._benchmark_legacy_profiler_step = True
        return {"bounded": True, "max_steps": steps, "source": "actual_torch_profiler_schedule_stepped_after_execute_model"}

    def benchmark_inspect_runtime(self):
        """Inspect already loaded worker objects without executing or allocating tensors."""
        import inspect
        def describe(value, methods):
            if value is None:
                return {"status": "unavailable"}
            result = {"class": type(value).__module__ + "." + type(value).__qualname__, "methods": {}}
            for name in methods:
                method = getattr(value, name, None)
                if callable(method):
                    try:
                        result["methods"][name] = {"signature": str(inspect.signature(method)),
                            "source_file": inspect.getsourcefile(method)}
                    except (TypeError, ValueError) as exc:
                        result["methods"][name] = {"inspection_error": type(exc).__name__}
            return result
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        modules = dict(model.named_modules()) if model is not None else {}
        candidates = {name: type(module).__module__ + "." + type(module).__qualname__
                      for name, module in modules.items() if any(term in name.lower()
                      for term in ("visual", "vision", "projector", "merger"))}
        return {"pid": os.getpid(), "runner": describe(runner, ("execute_model", "sample_tokens")),
                "model": describe(model, ("forward",)),
                "sampler": describe(getattr(runner, "sampler", None), ("forward",)),
                "model_sampler": describe(getattr(model, "sampler", None), ("forward",)),
                "candidate_modules": candidates,
                "scope": "actual loaded objects; signatures/module inventory, not proof hooks execute"}

    def benchmark_snapshot(self, reset="false"):
        import torch
        device = self.device
        if reset == "true":
            torch.cuda.reset_peak_memory_stats(device)
        memory = torch.cuda.memory_stats(device)
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        weight_bytes = None
        if model is not None:
            # Deduplicate tied parameter storages.
            storages = {}
            for p in list(model.parameters()) + list(model.buffers()):
                if p.is_cuda:
                    storage = p.untyped_storage()
                    storages[storage.data_ptr()] = storage.nbytes()
            weight_bytes = sum(storages.values())
        kv_bytes = None
        caches = getattr(runner, "kv_caches", None)
        if caches is not None:
            storages = {}
            def visit(value):
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    storage = value.untyped_storage()
                    storages[storage.data_ptr()] = storage.nbytes()
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        visit(item)
            visit(caches)
            kv_bytes = sum(storages.values()) if storages else None
        config = getattr(self, "vllm_config", None)
        model_config = getattr(self, "model_config", None)
        if model_config is None and config is not None:
            model_config = getattr(config, "model_config", None)
        compile_config = getattr(config, "compilation_config", None)
        pinned = None
        host_stats = getattr(getattr(torch.cuda, "memory", None), "host_memory_stats", None)
        if callable(host_stats):
            try:
                host = host_stats()
                pinned = host.get("allocated_bytes.all.current", host.get("allocated_bytes.current"))
            except (RuntimeError, AttributeError):
                pass
        attention = getattr(runner, "attn_backend", None)
        groups = getattr(runner, "attn_groups", None)
        if attention is None and isinstance(groups, (list, tuple)):
            attention = [str(getattr(group, "backend", "unknown")) for group in groups]
        return {"pid": os.getpid(), "device": str(device), "scope": "worker_run_wide",
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "inactive_split_bytes": memory.get("inactive_split_bytes.all.current"),
                "inactive_split_to_reserved_ratio": memory.get("inactive_split_bytes.all.current", 0) /
                    max(torch.cuda.memory_reserved(device), 1),
                "allocator_oom_count_cumulative": memory.get("num_ooms"),
                "model_weight_memory_bytes": weight_bytes, "kv_cache_memory_bytes": kv_bytes,
                "activation_memory_bytes": None, "input_image_memory_bytes": None,
                "pinned_memory_bytes": pinned,
                "model_revision": getattr(model_config, "revision", None),
                "model_commit_hash": getattr(getattr(model_config, "hf_config", None), "_commit_hash", None),
                "tokenizer_revision": getattr(model_config, "tokenizer_revision", None),
                "dtype": str(getattr(model_config, "dtype", None)),
                "max_model_len": getattr(model_config, "max_model_len", None),
                "cuda_graph_mode": str(getattr(compile_config, "cudagraph_mode", "unknown")),
                "compilation_config": str(compile_config),
                "cache_config": str(getattr(config, "cache_config", None)),
                "kv_cache_num_gpu_blocks": getattr(getattr(config, "cache_config", None), "num_gpu_blocks", None),
                "kv_cache_block_size": getattr(getattr(config, "cache_config", None), "block_size", None),
                "scheduler_config": str(getattr(config, "scheduler_config", None)),
                "attention_backend": str(attention) if attention is not None else "unknown",
                "unavailable_reason": "activation/input memory require allocation attribution; pinned stats only when allocator exposes them; never allocated-minus-weights"}

    def benchmark_telemetry(self, enable="true"):
        """Collect already-host-resident V1 input/output IDs without device reads or syncs."""
        runner = self.model_runner
        if enable != "true":
            original = getattr(self, "_benchmark_telemetry_original", None)
            if original is not None:
                runner.execute_model = original
                self._benchmark_telemetry_original = None
            return {"requests": getattr(self, "_benchmark_requests", {}),
                    "steps": getattr(self, "_benchmark_steps", {}),
                    "output_ids_scope": "only already-materialized Python sampled_token_ids; async outputs are not synchronized"}
        if getattr(self, "_benchmark_telemetry_original", None) is not None:
            raise RuntimeError("telemetry already enabled")
        if not hasattr(runner, "execute_model"):
            return {"status": "unavailable", "reason": "no V1 execute_model"}
        self._benchmark_requests = {}
        self._benchmark_steps = {"step_count": 0, "batch_size_step_counts": {},
                                 "scheduled_tokens_step_counts": {}, "request_batch_size_histograms": {}, "observed_schemas": {}, "output_type_step_counts": {}}
        original = runner.execute_model
        self._benchmark_telemetry_original = original
        @functools.wraps(original)
        def execute(scheduler_output, *args, **kwargs):
            for request in getattr(scheduler_output, "scheduled_new_reqs", []):
                ids = getattr(request, "prompt_token_ids", None)
                positions = []
                for feature in getattr(request, "mm_features", []):
                    position = getattr(feature, "mm_position", None)
                    length = getattr(position, "length", None)
                    offset = getattr(position, "offset", None)
                    if isinstance(length, int) and isinstance(offset, int):
                        positions.append({"offset": offset, "length": length,
                                          "modality": getattr(feature, "modality", None)})
                old = self._benchmark_requests.get(request.req_id, {})
                self._benchmark_requests[request.req_id] = {
                    "prompt_token_ids": ids if isinstance(ids, list) else None,
                    "multimodal_positions": positions,
                    "generated_token_ids": old.get("generated_token_ids", []),
                    "output_ids_observed_steps": old.get("output_ids_observed_steps", 0)}
            schemas = self._benchmark_steps["observed_schemas"]
            if "scheduler_output" not in schemas:
                schemas["scheduler_output"] = {"class": type(scheduler_output).__module__ + "." + type(scheduler_output).__qualname__,
                    "required_attributes": {name: hasattr(scheduler_output, name) for name in
                        ("scheduled_new_reqs", "scheduled_cached_reqs", "num_scheduled_tokens", "total_num_scheduled_tokens")}}
            new = getattr(scheduler_output, "scheduled_new_reqs", [])
            if new and "new_request" not in schemas:
                schemas["new_request"] = {name: hasattr(new[0], name) for name in
                    ("req_id", "prompt_token_ids", "num_computed_tokens", "mm_features")}
            result = original(scheduler_output, *args, **kwargs)
            output_type = type(result).__module__ + "." + type(result).__qualname__
            counts = self._benchmark_steps["output_type_step_counts"]
            counts[output_type] = counts.get(output_type, 0) + 1
            scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
            batch_size = len(scheduled)
            telemetry = self._benchmark_steps
            telemetry["step_count"] += 1
            for key, value in (("batch_size_step_counts", batch_size),
                               ("scheduled_tokens_step_counts", getattr(scheduler_output, "total_num_scheduled_tokens", 0))):
                histogram = telemetry[key]
                histogram[value] = histogram.get(value, 0) + 1
            for rid in scheduled:
                histogram = telemetry["request_batch_size_histograms"].setdefault(rid, {})
                histogram[batch_size] = histogram.get(batch_size, 0) + 1
            ids = getattr(result, "sampled_token_ids", None)
            req_ids = getattr(result, "req_ids", None)
            if isinstance(ids, list) and isinstance(req_ids, list):
                for rid, tokens in zip(req_ids, ids):
                    if rid in self._benchmark_requests and isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                        self._benchmark_requests[rid]["generated_token_ids"].extend(tokens)
                        self._benchmark_requests[rid]["output_ids_observed_steps"] += 1
            return result
        runner.execute_model = execute
        return {"status": "enabled", "scope": "actual scheduler inputs and host-ready outputs; no CUDA synchronization"}

    def benchmark_effective_limit(self, prompt_tokens, requested="null"):
        """Use the installed server helper, not a guessed context subtraction."""
        config = getattr(self, "model_config", None)
        if config is None:
            config = self.vllm_config.model_config
        for module_name in ("vllm.entrypoints.serve.utils.api_utils",
                            "vllm.entrypoints.openai.serving_engine",
                            "vllm.entrypoints.utils"):
            try:
                helper = getattr(importlib.import_module(module_name), "get_max_tokens")
            except (ImportError, AttributeError):
                continue
            import inspect
            parameters = inspect.signature(helper).parameters
            kwargs = {"max_model_len": config.max_model_len,
                      "max_tokens": json.loads(requested), "input_length": int(prompt_tokens),
                      "default_sampling_params": config.get_diff_sampling_param(),
                      "override_max_tokens": getattr(config, "override_generation_config", {}).get("max_new_tokens")}
            if "request" in parameters:
                try:
                    protocol = importlib.import_module("vllm.entrypoints.openai.protocol")
                    kwargs["request"] = protocol.ChatCompletionRequest(
                        model="benchmark", messages=[{"role": "user", "content": ""}], max_tokens=json.loads(requested))
                except (ImportError, AttributeError, TypeError):
                    continue
            kwargs = {k: v for k, v in kwargs.items() if k in parameters}
            required = {name for name, p in parameters.items() if p.default is inspect.Parameter.empty
                        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
            if not required <= set(kwargs):
                continue
            return {"effective_max_tokens": helper(**kwargs), "source": module_name + ".get_max_tokens"}
        return {"effective_max_tokens": None, "reason": "installed server limit helper unsupported"}

    def benchmark_profile_ranges(self, enable="true", module_map="{}"):
        """Annotate actual forward calls; hooks are only active during representative profiling.

        CUDA Graph replay may not re-enter Python. Missing ranges remain unavailable.
        Explicit module paths can override automatic, recorded candidate selection.
        """
        import torch
        if enable != "true":
            for handle in getattr(self, "_benchmark_handles", []):
                handle.remove()
            self._benchmark_handles = []
            for owner, name, original in getattr(self, "_benchmark_wrapped", []):
                setattr(owner, name, original)
            self._benchmark_wrapped = []
            return {"disabled": True}
        if getattr(self, "_benchmark_handles", None) or getattr(self, "_benchmark_wrapped", None):
            raise RuntimeError("benchmark ranges already installed")
        model = self.model_runner.model
        modules = dict(model.named_modules())
        mapping = json.loads(module_map)
        candidates = {"vision_encoder": ["visual", "vision_model", "vision_tower", "model.visual", "model.vision_model"],
                      "projector": ["multi_modal_projector", "mm_projector", "projector", "visual.merger",
                                    "model.multi_modal_projector", "model.visual.merger"]}
        def make_wrapper(fn, label):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                with self._benchmark_range(label):
                    return fn(*args, **kwargs)
            return wrapped
        installed = {}
        self._benchmark_handles = []
        self._benchmark_wrapped = []
        for stage, names in candidates.items():
            path = mapping.get(stage)
            if path is None:
                path = next((n for n in names if n in modules), None)
            if path is None:
                continue
            if path not in modules:
                raise ValueError(f"unknown module path {path}")
            module = modules[path]
            # Wrap forward, preserving the module and its registered hooks.
            original = module.forward
            module.forward = make_wrapper(original, stage)
            self._benchmark_wrapped.append((module, "forward", original))
            installed[stage] = {"module_path": path, "class": type(module).__name__}
        runner = self.model_runner
        if hasattr(runner, "execute_model"):
            execute_original = runner.execute_model
            @functools.wraps(execute_original)
            def execute(scheduler_output, *args, **kwargs):
                # Classify V1 from explicit scheduler prompt lengths/computed positions,
                # not from the unreliable assumption that one scheduled token means decode.
                kinds = []
                states = {}
                for request in getattr(scheduler_output, "scheduled_new_reqs", []):
                    ids = getattr(request, "prompt_token_ids", None)
                    computed = getattr(request, "num_computed_tokens", None)
                    if ids is not None and computed is not None:
                        states[request.req_id] = (len(ids), computed)
                cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
                if cached is not None:
                    requests = getattr(runner, "requests", {})
                    for rid, computed in zip(getattr(cached, "req_ids", []), getattr(cached, "num_computed_tokens", [])):
                        request = requests.get(rid)
                        prompt_len = getattr(request, "num_prompt_tokens", None)
                        if prompt_len is None:
                            ids = getattr(request, "prompt_token_ids", None)
                            prompt_len = len(ids) if ids is not None else None
                        if prompt_len is not None:
                            states[rid] = (prompt_len, computed)
                scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
                for rid in scheduled:
                    if rid in states:
                        prompt_len, computed = states[rid]
                        kinds.append("prefill" if computed < prompt_len else "decode")
                    else:
                        kinds.append("unknown")
                label = kinds[0] if kinds and len(set(kinds)) == 1 and kinds[0] != "unknown" else "mixed_or_unknown_step"
                previous_phase = getattr(self, "_benchmark_current_phase", None)
                self._benchmark_current_phase = label
                self._benchmark_step_finishes_prompt = label == "prefill" and bool(scheduled) and all(
                    rid in states and states[rid][1] + count >= states[rid][0] for rid, count in scheduled.items())
                try:
                    with self._benchmark_range("runner_" + label, annotation=label):
                        result = execute_original(scheduler_output, *args, **kwargs)
                finally:
                    self._benchmark_current_phase = previous_phase
                if getattr(self, "_benchmark_legacy_profiler_step", False):
                    self.profiler.step()
                return result
            runner.execute_model = execute
            self._benchmark_wrapped.append((runner, "execute_model", execute_original))
            installed["runner_phases"] = "scheduler prompt length versus actual computed position; graph replay remains inside range"
        sampler = getattr(runner, "sampler", None)
        if sampler is None:
            sampler = getattr(model, "sampler", None)
        if sampler is not None and hasattr(sampler, "forward"):
            sampler_original = sampler.forward
            sampler.forward = make_wrapper(sampler_original, "sampling")
            self._benchmark_wrapped.append((sampler, "forward", sampler_original))
            installed["sampling"] = type(sampler).__name__
        # V0 attention metadata exposes explicit prefill/decode token counts.
        # V1 variants without these fields deliberately get no fabricated phase label.
        original = model.forward
        @functools.wraps(original)
        def forward(*args, **kwargs):
            phase = getattr(self, "_benchmark_current_phase", None)
            stage = phase if phase in ("prefill", "decode") else "model_forward_unclassified"
            try:
                from vllm.forward_context import get_forward_context
                metadata = get_forward_context().attn_metadata
                if isinstance(metadata, dict):
                    metadata = next(iter(metadata.values()), None)
                prefill = getattr(metadata, "num_prefill_tokens", None)
                decode = getattr(metadata, "num_decode_tokens", None)
                if isinstance(prefill, int) and isinstance(decode, int):
                    if prefill > 0 and decode == 0:
                        stage = "prefill"
                    elif decode > 0 and prefill == 0:
                        stage = "decode"
                    else:
                        stage = "mixed_prefill_decode"
            except (AssertionError, AttributeError, RuntimeError, ImportError):
                pass
            with self._benchmark_range(stage):
                return original(*args, **kwargs)
        model.forward = forward
        self._benchmark_wrapped.append((model, "forward", original))
        return {"installed": installed, "model_modules": list(modules),
                "scope": "representative_profile_only", "phase_source": "actual attention metadata",
                "warning": "graph replay may bypass Python ranges; parent vision ranges can include projector"}
