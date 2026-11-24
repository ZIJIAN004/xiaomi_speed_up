"""Single-GPU AsyncLLM adapter. No HTTP or per-output CUDA synchronization."""
from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import io
import json
import os
import threading
import time

_ACTIVE = None
BASE_URL = "offline://single"


def active():
    if _ACTIVE is None:
        raise RuntimeError("offline engine is not initialized")
    return _ACTIVE


async def collect_stream(outputs, request_id, clock=time.perf_counter):
    """Actual DELTA token IDs; grouped output is not fabricated token ITL."""
    text, ids, events = [], [], []
    prompt_ids, finish, stop_reason = None, None, None
    finished = False
    async for output in outputs:
        if output.prompt_token_ids is not None:
            prompt_ids = list(output.prompt_token_ids)
        if len(output.outputs) > 1:
            raise ValueError("offline benchmark requires exactly one completion")
        for completion in output.outputs:
            delta_ids = list(completion.token_ids)
            if delta_ids:
                events.append({"time_s": clock(), "token_count": len(delta_ids),
                               "has_visible_text": bool(completion.text)})
                ids.extend(delta_ids)
            text.append(completion.text)
            finish = completion.finish_reason or finish
            if completion.stop_reason is not None:
                stop_reason = completion.stop_reason
        finished = output.finished
    if not finished or finish is None:
        raise RuntimeError("offline stream ended without a terminal completion")
    times = [e["time_s"] for e in events]
    intervals = [(b-a)*1000 for a,b in zip(times, times[1:])]
    precise = bool(events) and all(e["token_count"] == 1 for e in events)
    return {"text": "".join(text), "events": events, "finish_reason": finish,
            "stop_reason": stop_reason, "generated_token_ids": ids, "prompt_token_ids": prompt_ids,
            "usage": {"completion_tokens": len(ids), "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None},
            "server_request_id": request_id, "first_content_time_s": times[0] if times else None,
            "last_content_time_s": times[-1] if times else None,
            "chunk_interval_ms": intervals, "itl_ms": intervals if precise else None,
            "itl_source": "one_actual_token_id_per_offline_delta" if precise else None,
            "itl_unavailable_reason": None if precise else "engine delivered grouped token IDs; no invented token timestamps"}


class OfflineBackend:
    def __init__(self, args):
        if _ACTIVE is not None:
            raise RuntimeError("only one offline GPU engine is permitted")
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name="offline-engine-loop", daemon=True)
        self.engine = None
        self.thread.start()
        try:
            self.call(self._initialize(), timeout=1200)
        except BaseException:
            self.close()
            raise

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coroutine, timeout=600):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("offline engine operation timed out; cancellation initiated")

    async def _initialize(self):
        from baseline.harness.server_processor import install
        install()
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.config import ProfilerConfig
        config = {"profiler": "torch", "torch_profiler_dir": str(self.args.profiler_dir),
                  # Python stacks are not used by trace_analysis and dominate trace size/export time.
                  "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                  "torch_profiler_with_memory": True, "torch_profiler_with_flops": True}
        fields = set(getattr(ProfilerConfig, "__dataclass_fields__", {})) | set(getattr(ProfilerConfig, "__annotations__", {}))
        if "max_iterations" in fields:
            config["max_iterations"] = self.args.profile_max_iterations
        if "ignore_frontend" in fields:
            config["ignore_frontend"] = True
        self.processor = AutoProcessor.from_pretrained(self.args.model, trust_remote_code=self.args.trust_remote_code)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=self.args.model, served_model_name=self.args.served_model_name,
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            max_model_len=self.args.max_model_len,
            dtype=self.args.dtype, gpu_memory_utilization=self.args.gpu_memory_utilization,
            trust_remote_code=self.args.trust_remote_code, generation_config="vllm",
            enforce_eager=False,  # vLLM default: CUDA Graph and torch.compile enabled
            # Only experimental variable across models: speculative decoding (e.g. DSpark); None = plain decode.
            speculative_config=json.loads(self.args.speculative_config) if self.args.speculative_config else None,
            worker_extension_cls="baseline.harness.worker_extension.BenchmarkWorkerExtension",
            profiler_config=ProfilerConfig(**config), disable_log_stats=False))
        parallel = self.engine.vllm_config.parallel_config
        if any(getattr(parallel, name) != 1 for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")):
            raise RuntimeError("offline benchmark requires actual TP=PP=DP=1")
        tokenizer = self.processor.tokenizer
        self.metadata = {"backend": "offline", "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                         "data_parallel_size": 1, "max_inflight_requests": self.args.workers,
                         "scheduler_max_num_seqs": self.engine.vllm_config.scheduler_config.max_num_seqs,
                         "padding_side": tokenizer.padding_side, "max_model_len": self.args.max_model_len,
                         "prompt_policy": "native AutoProcessor template; tokenize=False, enable_thinking=False; engine expands image placeholders; add_special_tokens=False",
                         "eos_token_id": tokenizer.eos_token_id,
                         "generation_limit_policy": "max_tokens=None uses installed InputProcessor remaining context; no default 16-token cap"}

    def infer(self, image, args, request_id, attempt):
        return self.call(self._infer(image, args, request_id, attempt), timeout=args.request_timeout + 10)

    async def _infer(self, image, args, request_id, attempt):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from baseline.harness.processor_audit import PROMPT
        messages = [{"role": "user", "content": [{"type": "text", "text": getattr(args, "request_prompt", PROMPT)}, {"type": "image", "image": image}]}]
        begin = time.perf_counter()
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        template_ms = (time.perf_counter()-begin)*1000
        params = SamplingParams(temperature=0, top_p=0.8, repetition_penalty=1.0,
                                presence_penalty=args.retry_presence_penalty if attempt else 0.0,
                                max_tokens=args.max_tokens, output_kind=RequestOutputKind.DELTA,
                                logprobs=None if args.no_logprobs else 1)
        submitted = time.perf_counter()
        try:
            result = await asyncio.wait_for(collect_stream(self.engine.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image}}, params, request_id,
                tokenization_kwargs={"add_special_tokens": False}), request_id),
                timeout=args.request_timeout)
        except BaseException:
            await self.engine.abort(request_id)
            raise
        result.update(engine_submit_s=submitted, chat_template_time_ms=template_ms,
                      effective_max_tokens=args.max_tokens if args.max_tokens is not None else
                      (args.max_model_len-len(result["prompt_token_ids"]) if result["prompt_token_ids"] is not None else None))
        return result

    async def _rpc(self, method, args):
        return await self.engine.collective_rpc(method, timeout=600, args=tuple(args))

    def rpc(self, method, args=(), timeout=600):
        return self.call(self._rpc(method, args), timeout=timeout)

    async def _control(self, endpoint):
        method = getattr(self.engine, endpoint, None)
        if method is None:
            raise RuntimeError("installed offline engine lacks " + endpoint)
        value = method()
        return await value if inspect.isawaitable(value) else value

    def control(self, endpoint):
        value = self.call(self._control(endpoint))
        response = io.BytesIO(json.dumps({"result": value}, default=str).encode())
        response.status = 200
        return response

    def get(self, endpoint):
        if endpoint == "metrics":
            from prometheus_client import REGISTRY, CollectorRegistry, generate_latest, multiprocess
            registry = REGISTRY
            if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry).decode()
        if endpoint.startswith("server_info"):
            return json.dumps(self.metadata)
        if endpoint == "version":
            import importlib.metadata
            return json.dumps({"version": importlib.metadata.version("vllm"), "backend": "offline"})
        raise ValueError("unsupported offline metadata endpoint: " + endpoint)

    def close(self):
        async def shutdown():
            if self.engine is not None:
                value = self.engine.shutdown()
                if inspect.isawaitable(value):
                    await value
        try:
            self.call(shutdown(), timeout=60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)
            if not self.thread.is_alive():
                self.loop.close()

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError("only one offline GPU engine is permitted")
        _ACTIVE = self
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        try:
            self.close()
        finally:
            _ACTIVE = None
