#!/usr/bin/env python3
"""Streaming vLLM benchmark. Full-run metrics and representative profiling stay separate."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from pathlib import Path

from benchmark.metrics import PADDING_FIELDS, STAGES, TOKEN_FIELDS, cuda_event_statistics, dump_json, load_jsonl, stats, weighted_stats
from benchmark.monitor import Monitor, get_url, rpc
from benchmark.processor_audit import PROMPT, ProcessorAudit, prepare_image
from benchmark.output_storage import append_raw, finalize_predictions, retain_outputs


def post_json(url, payload=None, timeout=600):
    if url.startswith("offline://single/"):
        from benchmark.offline_backend import active
        return active().control(url.split("offline://single/", 1)[1])
    data = json.dumps(payload or {}).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def consume_sse(lines, clock=time.perf_counter):
    """Text chunks are not tokens. Precise host ITL requires exactly one logprob token/event."""
    content, events, finish, usage = [], [], None, None
    pending = []
    done = False
    server_id = None

    def consume(data):
        nonlocal finish, usage, done, server_id
        if data == "[DONE]":
            done = True
            return
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(json.dumps(event["error"], ensure_ascii=False))
        server_id = event.get("id", server_id)
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            logprobs = (choice.get("logprobs") or {}).get("content")
            # Also retain token events with no visible text, if exposed by logprobs.
            if text or logprobs:
                now = clock()
                content.append(text)
                events.append({"time_s": now, "token_count": len(logprobs) if logprobs is not None else None,
                               "has_visible_text": bool(text)})
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if pending:
                consume("\n".join(pending))
                pending = []
                if done:
                    break
            continue
        if line.startswith("data:"):
            pending.append(line[5:].lstrip(" "))
    if pending:
        consume("\n".join(pending))
    if not done:
        raise RuntimeError("stream ended without [DONE]; output is incomplete")
    times = [e["time_s"] for e in events]
    chunks = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    one_token = bool(events) and all(e["token_count"] == 1 for e in events)
    return {"text": "".join(content), "events": events, "finish_reason": finish,
            "usage": usage, "server_request_id": server_id, "first_content_time_s": times[0] if times else None,
            "last_content_time_s": times[-1] if times else None,
            "chunk_interval_ms": chunks, "itl_ms": chunks if one_token else None,
            "itl_source": "one_logprob_token_per_stream_event" if one_token else None,
            "itl_unavailable_reason": None if one_token else "stream events are not proven one-token events"}


def validate_manifest(records):
    seen = set()
    for r in records:
        sid = r.get("sample_id")
        if not isinstance(sid, str) or not sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sid):
            raise ValueError("sample_id must be a nonempty safe filename ID")
        if sid in seen:
            raise ValueError(f"duplicate sample_id: {sid}")
        seen.add(sid)
        if len(r.get("images", [])) != 1:
            raise ValueError(f"{sid}: exactly one image per request is required")
        if not Path(r["images"][0]).is_file():
            raise FileNotFoundError(r["images"][0])
        if r.get("num_pages", 1) != 1 or r.get("num_images", 1) != 1:
            raise ValueError(f"{sid}: this runner measures one page/image per request")


def request_one(sample, args, output, phase="steady", attempt=0, audit=None):
    start = time.perf_counter()
    record = {k: None for k in (*TOKEN_FIELDS, *PADDING_FIELDS)}
    record.update({stage + "_time_ms": None for stage in STAGES})
    record.update({k: sample.get(k) for k in ("document_id", "document_type", "has_table", "has_formula", "difficulty", "relative_path",
                  "request_unit", "page_sample_id", "region_index", "task_type", "bbox_2d", "regions_per_page", "prompt")})
    record.update(sample_id=sample["sample_id"], dataset=sample.get("dataset"), phase=phase,
                  attempt=attempt, image_path=sample["images"][0], num_images=1,
                  num_pages=None if sample.get("request_unit") == "region" else 1,
                  model=args.served_model_name, inference_framework="vllm", backend=getattr(args, "backend", "http"), client_workers=args.workers,
                  request_batch_size=1, actual_scheduler_batch_size=None,
                  requested_max_tokens=args.max_tokens, effective_max_tokens=None,
                  request_start_s=start, error=None, error_type=None, finish_reason=None,
                  success=False, output_path=None, peak_allocated_memory_gb=None,
                  peak_reserved_memory_gb=None, memory_scope="unavailable_per_request_under_concurrency",
                  stop_reason_is_eos=None, server_processor_time_ms=None,
                  gpu_ttft_ms=None, host_ttft_origin=("offline engine submission before input processing" if getattr(args, "backend", "http") == "offline" else "HTTP request submission after image serialization"))
    if audit:
        record.update(audit)
        # Audit timings are separate observations, not client or server stage times.
        record["processor_audit_time_ms"] = record.pop("processor_time_ms", None)
        record["processor_time_ms"] = None
    try:
        image, image_metrics = (prepare_image(sample["images"][0], bbox_2d=sample["bbox_2d"])
                                if "bbox_2d" in sample else prepare_image(sample["images"][0]))
        record.update(image_metrics)
        if image_metrics["source_frame_count"] != 1:
            image.close()
            raise ValueError("multipage/animated image must be split into page images before benchmarking")
        offline = getattr(args, "backend", "http") == "offline"
        if offline:
            from benchmark.offline_backend import active
            request_id = f"benchmark-{phase}-{sample['sample_id']}-{attempt}"
            try:
                inference_args = copy(args)
                inference_args.request_prompt = sample.get("prompt", PROMPT)
                streamed = active().infer(image, inference_args, request_id, attempt)
            finally:
                image.close()
            http_start = streamed["engine_submit_s"]
            http_end = time.perf_counter()
            record.update(backend="offline", host_ttft_origin="offline engine submission before input processing",
                          serialization_time_ms=None, chat_template_time_ms=streamed["chat_template_time_ms"],
                          http_metrics_unavailable_reason="no HTTP, Base64, JSON or SSE transport in offline mode",
                          effective_max_tokens=streamed["effective_max_tokens"],
                          effective_max_tokens_source="actual offline SamplingParams and expanded input length",
                          offline_generated_token_ids=streamed["generated_token_ids"],
                          offline_prompt_token_ids=streamed["prompt_token_ids"])
        else:
            serial_start = time.perf_counter()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image.close()
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            payload = {"model": args.served_model_name,
                       "messages": [{"role": "user", "content": [{"type": "text", "text": sample.get("prompt", PROMPT)},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
                       "temperature": 0, "top_p": 0.8, "presence_penalty": 0.0,
                       "repetition_penalty": 1.0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False},
                       "logprobs": not args.no_logprobs}
            if args.max_tokens is not None:
                payload["max_tokens"] = args.max_tokens
            if attempt:
                payload["presence_penalty"] = args.retry_presence_penalty
            body = json.dumps(payload).encode()
            record["serialization_time_ms"] = (time.perf_counter() - serial_start) * 1000
            http_start = time.perf_counter()
            request = urllib.request.Request(args.base_url + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json",
                                                      "X-Request-Id": f"benchmark-{phase}-{sample['sample_id']}-{attempt}"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=args.request_timeout) as response:
                streamed = consume_sse(response)
            http_end = time.perf_counter()
        record.update(http_start_s=http_start, http_end_s=http_end,
                      request_latency_ms=(http_end - http_start) * 1000,
                      finish_reason=streamed["finish_reason"], server_request_id=streamed["server_request_id"],
                      streaming_event_count=len(streamed["events"]), chunk_interval_ms=streamed["chunk_interval_ms"],
                      itl_ms=streamed["itl_ms"], itl_source=streamed["itl_source"],
                      itl_unavailable_reason=streamed["itl_unavailable_reason"])
        if offline:
            record["engine_submit_s"] = record.pop("http_start_s")
            record["engine_end_s"] = record.pop("http_end_s")
            record["offline_output_event_count"] = record.pop("streaming_event_count")
            record["host_output_interval_ms"] = record.pop("chunk_interval_ms")
        if record["finish_reason"] is not None and record["finish_reason"] != "stop":
            record["stop_reason_is_eos"] = False
        first, last = streamed["first_content_time_s"], streamed["last_content_time_s"]
        record["first_output_event_host_s"] = first
        record["last_output_event_host_s"] = last
        record["stream_logprob_token_count"] = sum(e["token_count"] for e in streamed["events"]) if all(e["token_count"] is not None for e in streamed["events"]) else None
        if offline:
            record["offline_stream_token_count"] = record.pop("stream_logprob_token_count")
        record["itl_population"] = ("actual offline delta token IDs, including terminal IDs when exposed" if offline else
                                    "observed one-token SSE events; terminal EOS may not be streamed")
        record["ttft_ms"] = (first - http_start) * 1000 if first is not None else None
        record["ttft_e2e_ms"] = (first - start) * 1000 if first is not None else None
        record["host_decode_span_ms"] = (last - first) * 1000 if first is not None else None
        usage = streamed["usage"] or {}
        record["usage"] = usage
        record["generated_tokens"] = usage.get("completion_tokens")
        record["actual_generated_tokens"] = usage.get("completion_tokens")
        record["total_input_tokens"] = usage.get("prompt_tokens")
        if record["total_input_tokens"] is not None and record["generated_tokens"] is not None:
            record["total_processed_tokens"] = record["total_input_tokens"] + record["generated_tokens"]
        if audit and record["total_input_tokens"] is not None:
            agree = audit.get("processor_unpadded_input_length") == record["total_input_tokens"]
            record["processor_server_input_length_match"] = agree
            record["processor_server_semantics_verified"] = False
            record["processor_verification_note"] = "length agreement is necessary, not proof of identical IDs/vision semantics"
            # Local processor observations remain explicitly scoped even when lengths agree.
            for field in ("visual_tokens", "prompt_text_tokens", "special_input_tokens", "image_patch_count"):
                record[field + "_source"] = "offline_processor_audit" if record.get(field) is not None else None
        n = record["generated_tokens"]
        record["time_per_generated_token_ms"] = record["request_latency_ms"] / n if n else None
        # Do not name host stream duration decode_time: it includes detokenization/network scheduling.
        record["host_tpot_after_first_ms"] = record["host_decode_span_ms"] / (n - 1) if n and n > 1 and first is not None else None
        record["tpot_ms"] = None
        intervals = record["itl_ms"]
        record["itl_statistics_ms"] = stats(intervals or [])
        record["itl_first_ms"] = intervals[0] if intervals else None
        record["itl_steady_statistics_ms"] = stats(intervals[1:] if intervals else [])
        text = streamed["text"]
        raw_write_start = time.perf_counter()
        append_raw(output, record, text)
        raw_write_ms = (time.perf_counter() - raw_write_start) * 1000
        record["raw_output_io_time_ms"] = raw_write_ms
        record["empty_output"] = not bool(text.strip())
        if not text.strip():
            raise ValueError("empty_output")
        post_start = time.perf_counter()
        post_cpu_start = time.thread_time()
        from postprocess.document import postprocess_e2e_markdown, _clean_truncated_repeats
        from postprocess.repetition_guard import analyze_repetition
        analysis = analyze_repetition(text, finish_reason=record["finish_reason"],
                                      same_character_run=128, min_repeats=8, min_repeated_chars=512,
                                      min_ratio=0.5, max_unit_chars=256)
        record["repetition_analysis"] = analysis
        processed = (text.strip() if sample.get("request_unit") == "region" else
                     postprocess_e2e_markdown(_clean_truncated_repeats(text), convert_otsl=True))
        record["postprocess_time_ms"] = (time.perf_counter() - post_start) * 1000
        record["postprocessing_cpu_time_ms"] = (time.thread_time() - post_cpu_start) * 1000
        record["postprocessed_empty_output"] = not bool(processed.strip())
        if not processed.strip():
            raise ValueError("postprocessed_empty_output")
        directory = output / phase / f"attempt_{attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (sample["sample_id"] + ".md")
        write_start = time.perf_counter()
        path.write_text(processed, encoding="utf-8")
        record["output_io_time_ms"] = raw_write_ms + (time.perf_counter() - write_start) * 1000
        record["output_io_scope"] = "shared raw JSONL encode/lock/append plus Markdown write; excludes final move and post-evaluation retention"
        record.update(output_path=str(path.resolve()), success=True)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:8000]
        record.update(error=f"HTTP {exc.code}: {detail}", error_type="http_error", http_status=exc.code)
    except Exception as exc:
        record.update(error=str(exc), error_type=type(exc).__name__)
    record["request_end_s"] = time.perf_counter()
    record["end_to_end_latency_ms"] = (record["request_end_s"] - start) * 1000
    # Terminal length failures are counted even if postprocessing failed.
    record["length_stopped"] = record.get("finish_reason") == "length"
    record["oom_error"] = bool(record["error"] and any(s in record["error"].lower() for s in ("out of memory", "cuda_oom", "outofmemory")))
    return record


def system_tokens_from_chatml(ids, tokenizer):
    """Parse only a verified native ChatML layout; unsupported templates return null."""
    vocab = tokenizer.get_vocab()
    start, end = vocab.get("<|im_start|>"), vocab.get("<|im_end|>")
    if start is None or end is None or not ids or ids[0] != start:
        return None
    prefixes = {role: tokenizer.encode(role + "\n", add_special_tokens=False)
                for role in ("system", "user", "assistant")}
    starts = [i for i, token in enumerate(ids) if token == start]
    special = set(tokenizer.all_special_ids)
    count = 0
    roles = []
    for offset in starts:
        matching = [role for role, prefix in prefixes.items()
                    if prefix and ids[offset + 1:offset + 1 + len(prefix)] == prefix]
        if len(matching) != 1:
            return None
        role = matching[0]
        roles.append(role)
        body = offset + 1 + len(prefixes[role])
        if role == "system":
            try:
                finish = ids.index(end, body)
            except ValueError:
                return None
            count += sum(token not in special for token in ids[body:finish])
    return count if "user" in roles else None


def reconcile_worker_records(records, telemetry, tokenizer=None):
    requests = {}
    steps = []
    for worker in telemetry.get("results", []):
        for rid, value in worker.get("requests", {}).items():
            # Tensor parallel replicas describe the same request; disagreement invalidates attribution.
            if rid in requests and requests[rid] != value:
                requests[rid] = {"conflict": True}
            else:
                requests[rid] = value
        steps.append(worker.get("steps", {}))
    for r in records:
        if tokenizer is not None and r.get("success"):
            final_text = Path(r["output_path"]).read_text(encoding="utf-8")
            r["useful_output_tokens"] = len(tokenizer.encode(final_text, add_special_tokens=False))
            r["useful_output_tokens_source"] = "actual tokenizer on accepted postprocessed output, not raw generated token IDs"
        output_ids = r.get("offline_generated_token_ids")
        if tokenizer is not None and output_ids and len(output_ids) == r.get("generated_tokens"):
            special = set(tokenizer.all_special_ids)
            r["generated_special_tokens"] = sum(t in special for t in output_ids)
            r["generated_visible_tokens"] = len(output_ids) - r["generated_special_tokens"]
            r["generated_token_classification_source"] = "actual_offline_delta_token_ids"
            r["stop_reason_is_eos"] = (True if output_ids[-1] == tokenizer.eos_token_id else None) if r.get("finish_reason") == "stop" else False
        server_id = r.get("server_request_id")
        candidates = [(rid, value) for rid, value in requests.items()
                      if server_id and (rid == server_id or rid.startswith(server_id + "-"))]
        if len(candidates) != 1 or candidates[0][1].get("conflict"):
            r["worker_input_unavailable_reason"] = "no unique server request-ID match in worker telemetry"
            continue
        rid, observed = candidates[0]
        ids = observed.get("prompt_token_ids")
        positions = observed.get("multimodal_positions", [])
        r["worker_request_id"] = rid
        # Do not multiply token-step populations by tensor-parallel replica count.
        histograms = [s.get("request_batch_size_histograms", {}).get(rid) for s in steps]
        histograms = [h for h in histograms if h]
        histogram = histograms[0] if histograms and all(h == histograms[0] for h in histograms) else {}
        r["actual_scheduler_batch_size_histogram"] = histogram
        r["actual_scheduler_batch_size"] = weighted_stats(histogram)
        if isinstance(ids, list) and len(ids) == r.get("total_input_tokens"):
            r.update(input_ids_length=len(ids), prompt_length_i=len(ids), processor_server_semantics_verified=False,
                     worker_input_length_verified=True)
            visual_indices = set()
            valid = bool(positions)
            for pos in positions:
                begin, length = pos["offset"], pos["length"]
                if begin < 0 or length < 0 or begin + length > len(ids):
                    valid = False
                    break
                span = set(range(begin, begin + length))
                if visual_indices & span:
                    valid = False
                    break
                visual_indices |= span
            if valid:
                r.update(visual_tokens=len(visual_indices), visual_length_i=len(visual_indices),
                         visual_tokens_source="actual_worker_multimodal_placeholder_spans")
            if tokenizer is not None and valid:
                special = set(tokenizer.all_special_ids)
                r["special_input_tokens"] = sum(i not in visual_indices and token in special for i, token in enumerate(ids))
                r["prompt_text_tokens"] = sum(i not in visual_indices and token not in special for i, token in enumerate(ids))
                r["prompt_text_tokens_source"] = "actual_worker_input_ids_excluding_visual_spans_and_special_ids"
                user_ids = tokenizer.encode(r.get("prompt") or PROMPT, add_special_tokens=False)
                matches = [i for i in range(len(ids) - len(user_ids) + 1) if ids[i:i + len(user_ids)] == user_ids]
                if len(matches) == 1:
                    r["user_text_tokens"] = len(user_ids)
                    r["template_or_implicit_system_text_tokens"] = r["prompt_text_tokens"] - len(user_ids)
                r["system_tokens"] = system_tokens_from_chatml(ids, tokenizer)
                residual = r.get("template_or_implicit_system_text_tokens")
                r["template_tokens"] = residual - r["system_tokens"] if residual is not None and r["system_tokens"] is not None and residual >= r["system_tokens"] else None
                r["template_decomposition_source"] = "verified_native_ChatML_role_prefix_IDs" if r["template_tokens"] is not None else None
                r["template_decomposition_unavailable_reason"] = "unsupported template layout; implicit system text is not guessed" if r["template_tokens"] is None else None
        output_ids = observed.get("generated_token_ids", [])
        if tokenizer is not None and output_ids and len(output_ids) == r.get("generated_tokens"):
            special = set(tokenizer.all_special_ids)
            r["generated_special_tokens"] = sum(t in special for t in output_ids)
            r["generated_visible_tokens"] = len(output_ids) - r["generated_special_tokens"]
            r["generated_token_classification_source"] = "actual_worker_sampled_ids_matched_to_server_usage"
            r["stop_reason_is_eos"] = (True if output_ids[-1] == tokenizer.eos_token_id else None) if r.get("finish_reason") == "stop" else False


def select_representatives(records, samples, count):
    """Cover size/output extremes and middle, then document structure/difficulty diversity."""
    by_id = {s["sample_id"]: s for s in samples}
    valid = sorted((r for r in records if r.get("success")), key=lambda r: r["sample_id"])
    features = {r["sample_id"]: set() for r in valid}
    for field in ("image_megapixels", "generated_tokens", "visual_tokens", "regions_per_page"):
        ordered = sorted((r for r in valid if r.get(field) is not None), key=lambda r: (r[field], r["sample_id"]))
        for i, r in enumerate(ordered):
            fraction = i / max(len(ordered) - 1, 1)
            label = "low" if fraction <= 0.2 else "high" if fraction >= 0.8 else "middle"
            features[r["sample_id"]].add((field, label))
    for r in valid:
        for field in ("has_table", "has_formula", "document_type", "difficulty"):
            if r.get(field) is not None:
                features[r["sample_id"]].add((field, json.dumps(r[field], sort_keys=True)))
    selected, covered = [], set()
    while len(selected) < count and valid:
        def score(r):
            novel = features[r["sample_id"]] - covered
            return sum(2 if k in ("image_megapixels", "generated_tokens", "visual_tokens", "has_table", "has_formula") else 1 for k, _ in novel)
        best = max(valid, key=score)
        selected.append(by_id[best["sample_id"]])
        covered |= features[best["sample_id"]]
        valid.remove(best)
    return selected


def safe_rpc(args, method, *values):
    try:
        return {"status": "measured", "results": rpc(args.base_url, method, values)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def profile_samples(args, samples, output, audit, monitor, tokenizer=None):
    records = []
    monitor.phase = "profile_setup"
    ranges = safe_rpc(args, "benchmark_profile_ranges", "true", args.module_map)
    dump_json(output / "profile_ranges.json", ranges)
    try:
        for index, sample in enumerate(samples):
            monitor.phase = "profile"
            destination = output / "profiles" / sample["sample_id"]
            destination.mkdir(parents=True, exist_ok=True)
            before = set(args.profiler_dir.rglob("*")) if args.profiler_dir else set()
            cache_resets = {}
            for endpoint in ("reset_prefix_cache", "reset_mm_cache", "reset_encoder_cache"):
                try:
                    with post_json(args.base_url + "/" + endpoint) as response:
                        body = response.read().decode("utf-8")
                        cache_resets[endpoint] = {"status": response.status, "response": body}
                except Exception as exc:
                    cache_resets[endpoint] = {"error": str(exc)}
            limit = safe_rpc(args, "benchmark_profile_limit", str(args.profile_max_iterations), str(args.profiler_dir))
            safe_rpc(args, "benchmark_telemetry", "true")
            event_begin = safe_rpc(args, "benchmark_event_timing", "true")
            memory_before = safe_rpc(args, "benchmark_snapshot", "true")
            started = False
            record = {"sample_id": sample["sample_id"], "phase": "profile", "success": False}
            try:
                bounds = limit.get("results", [])
                if not bounds or not all(b.get("bounded") for b in bounds):
                    raise RuntimeError("bounded worker profiler unavailable; inspect version, refusing unbounded trace memory growth")
                with post_json(args.base_url + "/start_profile"):
                    pass
                started = True
                record = request_one(sample, args, output, "profile", audit=audit.get(sample["sample_id"]))
            except Exception as exc:
                record["profile_error"] = str(exc)
            finally:
                if started:
                    try:
                        with post_json(args.base_url + "/stop_profile"):
                            pass
                    except Exception as exc:
                        record["profile_stop_error"] = str(exc)
                record["profile_limit"] = limit
                record["profile_limit_cleanup"] = safe_rpc(args, "benchmark_profile_limit", str(args.profile_max_iterations), str(args.profiler_dir), "true")
                event_data = safe_rpc(args, "benchmark_event_timing", "false")
                event_data["profiler_stop_succeeded"] = started and not bool(record.get("profile_stop_error"))
                dump_json(destination / "cuda_events.json", event_data)
                event_stats = cuda_event_statistics(event_data.get("results", []) if event_data["profiler_stop_succeeded"] else [])
                record["cuda_event_timing"] = event_stats
                record["cuda_event_setup"] = event_begin
                record["gpu_ttft_ms"] = event_stats["gpu_ttft_ms"]
                for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling"):
                    record[stage + "_time_ms"] = event_stats["stages"].get(stage, {}).get("exclusive_ms")
                    record[stage + "_cuda_event_time_ms"] = record[stage + "_time_ms"]
                    record[stage + "_wall_time_ms"] = event_stats.get("host_wall_stages", {}).get(stage, {}).get("inclusive_ms")
                    record[stage + "_wall_time_scope"] = "host wall span of stage event; not request wall time and not additive under overlap"
                    record[stage + "_time_source"] = "representative_current_stream_cuda_event_exclusive_span"
                    record[stage + "_cuda_event_time_source"] = "representative_current_stream_cuda_event_exclusive_span"
                    record[stage + "_wall_time_source"] = "representative_worker_host_wall_span"
                n = record.get("generated_tokens")
                record["tpot_ms"] = record["decode_time_ms"] / n if n and record.get("decode_time_ms") is not None else None
                model_ms = sum(event_stats["stages"][stage]["exclusive_ms"] for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling")
                               if stage in event_stats["stages"])
                record["model_only_generated_tokens_per_second"] = n * 1000 / model_ms if n and model_ms > 0 and all(stage in event_stats["stages"] for stage in ("vision_encoder", "projector", "prefill", "decode", "sampling")) else None
                for stage in ("prefill", "decode"):
                    record["runner_" + stage + "_step_time_ms"] = event_stats["stages"].get("runner_" + stage, {}).get("inclusive_ms")
                record["model_only_throughput_scope"] = "representative with profiler, sum exclusive observed model GPU stream spans; not steady throughput"
                record["worker_memory_before"] = memory_before
                record["worker_memory_after"] = safe_rpc(args, "benchmark_snapshot", "false")
                telemetry = safe_rpc(args, "benchmark_telemetry", "false")
                dump_json(destination / "worker_telemetry.json", telemetry)
                step_counts = [w.get("steps", {}).get("step_count") for w in telemetry.get("results", [])]
                caps = [b.get("max_steps") for b in limit.get("results", [])]
                record["trace_covers_full_request"] = bool(record.get("success") and event_data["profiler_stop_succeeded"]
                    and step_counts and caps and all(n is not None and n > 0 for n in step_counts)
                    and all(c is not None for c in caps) and max(step_counts) < min(caps))
                record["profile_trace_window"] = {"max_engine_iterations": caps, "observed_model_steps": step_counts,
                                                  "full_request": record["trace_covers_full_request"]}
                reconcile_worker_records([record], telemetry, tokenizer)
                snapshots = record["worker_memory_after"].get("results", [])
                for field, output_field in (("peak_allocated_bytes", "peak_allocated_memory_gb"), ("peak_reserved_bytes", "peak_reserved_memory_gb")):
                    values = [s.get(field) for s in snapshots]
                    record[output_field] = sum(values) / 1024 ** 3 if values and all(v is not None for v in values) else None
                record["memory_scope"] = "single representative request with profiler, includes resident weights/KV capacity"
            traces = []
            if args.profiler_dir:
                for path in sorted(set(args.profiler_dir.rglob("*")) - before):
                    if path.is_file():
                        # Keep one payload in profiler_raw; relative paths survive
                        # moving the complete run directory to another machine.
                        traces.append(os.path.relpath(path.resolve(), output.resolve()))
            dump_json(destination / "trace_reference.json", {
                "trace_files": traces, "relative_to": "dataset run directory",
                "storage": "single canonical payload in profiler_raw; no per-sample copy"})
            record["cache_resets"] = cache_resets
            record["profile_cache_scope"] = "cache-reset diagnostic rerun, not steady-state cache behavior"
            record["trace_files"] = traces
            record["profile_capture_complete"] = bool(traces) and not record.get("profile_stop_error") and started
            records.append(record)
            print(f"profile {index + 1}/{len(samples)}: {sample['sample_id']} traces={len(traces)}", flush=True)
    finally:
        dump_json(output / "profile_ranges_cleanup.json", safe_rpc(args, "benchmark_profile_ranges", "false"))
    return records


def environment():
    packages = {}
    for name in ("torch", "vllm", "transformers", "tokenizers", "pillow", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "platform": platform.platform(), "logical_cpu_count": os.cpu_count(), "packages": packages}
    for label, command in (("nvidia_smi", ["nvidia-smi"]), ("nvcc", ["nvcc", "--version"]),
                           ("git_revision", ["git", "rev-parse", "HEAD"])):
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=15)
            result[label] = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        except (OSError, subprocess.SubprocessError) as exc:
            result[label] = {"error": str(exc)}
    try:
        import torch
        result["torch_cuda_build_version"] = torch.version.cuda
    except ImportError:
        result["torch_cuda_build_version"] = None
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="Xiaomi-OCR-0")
    parser.add_argument("--backend", choices=("offline", "http"), default="offline")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--base-url", default="http://127.0.0.1:26000")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--representatives", type=int, default=50)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--monitor-interval", type=float, default=1)
    parser.add_argument("--dcgm-url", help="localhost DCGM exporter /metrics URL; optional hardware counters")
    parser.add_argument("--gpu-cost-per-hour", type=float, help="optional reserved-GPU price in a declared currency/hour")
    parser.add_argument("--long-document-pages", type=int, default=10)
    parser.add_argument("--server-processor-dir", type=Path, help="actual server processor JSONL directory from launcher")
    parser.add_argument("--profiler-dir", type=Path)
    parser.add_argument("--profile-max-iterations", type=int, default=64)
    parser.add_argument("--module-map", default="{}")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-processor-audit", action="store_true")
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--retry-repetitive", action="store_true")
    parser.add_argument("--retry-presence-penalty", type=float, default=1.5)
    parser.add_argument("--quality-command", help="JSON argv list with {manifest}, {predictions}, {output} placeholders")
    parser.add_argument("--output-retention", choices=("sampled", "all"), default="sampled")
    parser.add_argument("--retain-samples", type=int, default=10)
    parser.add_argument("--retain-worst", type=int, default=10)
    parser.add_argument("--retention-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.retain_samples < 0 or args.retain_worst < 0:
        parser.error("retention sample counts must be nonnegative")
    if args.workers < 1 or args.warmup < 3 or not 3 <= args.representatives <= 1355 or args.monitor_interval <= 0:
        parser.error("workers>=1, warmup>=3, representatives=3..1355 and monitor-interval>0 required")
    if args.profile_max_iterations < 1:
        parser.error("profile-max-iterations must be positive; unbounded representative traces are not enabled")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max-tokens must be positive or omitted")
    if args.output.exists():
        parser.error("output must be a new directory; refusing to reuse historical predictions")
    if args.backend == "http" and not args.base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("benchmark server and worker RPC must be bound to localhost")
    if args.dcgm_url and not args.dcgm_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("DCGM telemetry must use a localhost endpoint")
    if args.long_document_pages < 1 or (args.gpu_cost_per_hour is not None and args.gpu_cost_per_hour < 0):
        parser.error("long-document-pages must be positive and price cannot be negative")
    if not 0 < args.gpu_memory_utilization <= 1 or args.max_model_len < 1:
        parser.error("GPU memory utilization must be in (0,1] and max-model-len positive")
    if args.backend == "offline" and not args.gpu.isdecimal():
        parser.error("offline mode requires one numeric GPU index")
    if args.backend == "offline":
        args.base_url = "offline://single"
    return args


def main(args=None):
    args = args or parse_args()
    samples = load_jsonl(args.manifest)
    validate_manifest(samples)
    if not samples:
        raise ValueError("empty benchmark manifest")
    if getattr(args, "backend", "http") == "offline":
        from benchmark.offline_backend import _ACTIVE, OfflineBackend
        if _ACTIVE is None:
            if not args.gpu.isdecimal():
                raise ValueError("offline backend requires one GPU index")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            bootstrap = str(Path(__file__).resolve().parent / "server_bootstrap")
            os.environ["PYTHONPATH"] = bootstrap + os.pathsep + str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
            if args.profiler_dir is None:
                args.profiler_dir = args.output.resolve().parent / (args.output.name + "_profiler_raw")
            args.profiler_dir.mkdir(parents=True, exist_ok=True)
            if args.server_processor_dir is None:
                args.server_processor_dir = args.output.resolve().parent / (args.output.name + "_processor_raw")
            args.server_processor_dir.mkdir(parents=True, exist_ok=True)
            os.environ["VLLM_BENCHMARK_PROCESSOR_DIR"] = str(args.server_processor_dir.resolve())
            args.server_pid = os.getpid()
            with OfflineBackend(args):
                return main(args)
    output = args.output.resolve()
    output.mkdir(parents=True)
    shutil.copy2(args.manifest, output / "manifest.jsonl")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(benchmark_version="1.5" if samples[0].get("dataset") == "omnidocbench" else None,
                  request_unit=samples[0].get("request_unit", "page"), diagnostic_only=os.environ.get("BENCHMARK_RUNTIME_SMOKE") == "1", request_batch_size=1, actual_scheduler_batch_size=None, temperature=0,
                  top_p=0.8, repetition_penalty=1.0, dtype_policy="fixed server launch dtype",
                  streaming=True, http_client="stdlib urllib; no hidden retries, one connection per request",
                  max_tokens_policy="omitted by default; server generation-config=vllm",
                  profiler_results_excluded_from_steady_state=True,
                  client_timing_scope="file read through result write; queue wait before worker start excluded")
    if getattr(args, "backend", "http") == "offline":
        from benchmark.offline_backend import active
        config.update(http_client=None, transport="direct AsyncLLM Python API", tensor_parallel_size=1,
                      pipeline_parallel_size=1, data_parallel_size=1,
                      max_tokens_policy="explicit cap or remaining context via installed InputProcessor",
                      client_timing_scope="file read through result write; TTFT from offline engine submission",
                      offline_engine=active().metadata)
    config["environment"] = environment()
    config["environment"]["benchmark_version"] = config["benchmark_version"]
    (output / "environment.txt").write_text(json.dumps(config["environment"], indent=2) + "\n", encoding="utf-8")
    config["worker_initial"] = safe_rpc(args, "benchmark_snapshot", "false")
    for endpoint in ("server_info?config_format=json", "version"):
        try:
            config[endpoint.split("?")[0]] = json.loads(get_url(args.base_url + "/" + endpoint))
        except Exception as exc:
            config[endpoint.split("?")[0]] = {"status": "unavailable", "reason": str(exc)}
    dump_json(output / "config.json", config)
    if getattr(args, "backend", "http") == "offline":
        dump_json(output / "metric_migration.json", {
            "backend": "offline", "http_required": False,
            "retained": ["latency", "TTFT", "actual token IDs/counts", "throughput", "processor calls",
                         "worker scheduler telemetry", "CUDA Events", "bounded profiler", "memory", "quality"],
            "changed_scope": {"ttft_ms": "offline engine submission to first output token ID, not HTTP TTFT",
                              "itl_ms": "only when every nonempty delta contains one actual token ID",
                              "cold_start_latency_s": "offline engine initialization, not HTTP readiness",
                              "cpu_memory": "client/engine frontend share one PID; do not add their RSS/CPU figures",
                              "vllm_metrics": "native in-process Prometheus registry, no HTTP exporter"},
            "not_applicable": ["HTTP status", "HTTP transport latency", "Base64/JSON serialization", "SSE chunk intervals"],
            "validation": "runtime evidence in metric_coverage.csv; retained instrumentation is not proof of observed GPU measurements"})
    audit = {}
    tokenizer = None
    if getattr(args, "backend", "http") == "offline":
        from benchmark.offline_backend import active
        tokenizer = active().processor.tokenizer
    if not args.skip_processor_audit:
        try:
            inspector = ProcessorAudit(args.model, args.trust_remote_code)
            tokenizer = inspector.tokenizer
            dump_json(output / "processor_config.json", inspector.describe())
            for sample in samples:
                sid = sample["sample_id"]
                try:
                    image, _ = (prepare_image(sample["images"][0], bbox_2d=sample["bbox_2d"])
                                if "bbox_2d" in sample else prepare_image(sample["images"][0]))
                    try:
                        audit[sid] = (inspector.inspect(image, prompt=sample["prompt"])
                                      if "prompt" in sample else inspector.inspect(image))
                    finally:
                        image.close()
                except Exception as exc:
                    audit[sid] = {"processor_audit_error": str(exc)}
            del inspector
        except Exception as exc:
            dump_json(output / "processor_config.json", {"status": "unavailable", "reason": str(exc)})
    dump_json(output / "processor_audit.json", audit)
    records = []
    warmups = []
    profiles = []
    run_start = run_end = None
    with Monitor(output / "monitor", args.base_url, args.gpu, args.server_pid, args.monitor_interval, args.dcgm_url) as monitor:
        monitor.phase = "warmup"
        for i in range(args.warmup):
            r = request_one(samples[i % len(samples)], args, output, "warmup", attempt=i)
            warmups.append(r)
            print(f"warmup {i + 1}/{args.warmup}: success={r['success']}", flush=True)
        dump_json(output / "warmup.json", warmups)
        if not all(r["success"] for r in warmups):
            dump_json(output / "run_status.json", {"status": "failed_warmup", "reason": "inspect warmup.json; no steady-state results generated"})
            raise RuntimeError("warmup failed; refusing to publish invalid steady-state benchmark")
        telemetry_begin = safe_rpc(args, "benchmark_telemetry", "true")
        memory_before = safe_rpc(args, "benchmark_snapshot", "true")
        try:
            (output / "metrics_before.prom").write_text(get_url(args.base_url + "/metrics"), encoding="utf-8")
        except Exception as exc:
            config["metrics_before_error"] = str(exc)
        monitor.phase = "steady"
        run_start = time.perf_counter()
        with (output / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(request_one, s, args, output, "steady", 0, audit.get(s["sample_id"])) for s in samples]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    print(f"steady {len(records)}/{len(samples)}: {record['sample_id']} success={record['success']}", flush=True)
        run_end = time.perf_counter()
        monitor.phase = "steady_finalize"
        memory_after = safe_rpc(args, "benchmark_snapshot", "false")
        telemetry = safe_rpc(args, "benchmark_telemetry", "false")
        dump_json(output / "worker_telemetry.json", telemetry)
        reconcile_worker_records(records, telemetry, tokenizer)
        try:
            (output / "metrics_after.prom").write_text(get_url(args.base_url + "/metrics"), encoding="utf-8")
        except Exception as exc:
            config["metrics_after_error"] = str(exc)
        # Retries are a separate workload, excluded from primary steady-state throughput.
        final = {r["sample_id"]: r for r in records}
        retries = []
        monitor.phase = "retry"
        retry_start = time.perf_counter()
        if args.retry_repetitive:
            by_id = {s["sample_id"]: s for s in samples}
            targets = [r for r in records if r["success"] and r.get("repetition_analysis", {}).get("suspicious")]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(request_one, by_id[r["sample_id"]], args, output, "retry", 1,
                                       audit.get(r["sample_id"])) for r in targets]
                for future in as_completed(futures):
                    r = future.result()
                    r["adopted"] = bool(r["success"] and not r.get("repetition_analysis", {}).get("suspicious"))
                    if r["adopted"]:
                        final[r["sample_id"]] = r
                    retries.append(r)
        retry_end = time.perf_counter()
        mapping = finalize_predictions(output, samples, final)
        dump_json(output / "retry_metrics.json", retries)
        dump_json(output / "prediction_mapping.json", mapping)
        selected = select_representatives(records, samples, args.representatives)
        dump_json(output / "representative_selection.json", {"samples": selected,
                  "requested_count": args.representatives, "actual_count": len(selected),
                  "method": "weighted greedy coverage of observed low/middle/high area/output/visual lengths and supplied metadata",
                  "available_metadata_counts": {field: sum(r.get(field) is not None for r in records) for field in ("has_table", "has_formula", "document_type", "difficulty")},
                  "note": "missing structure labels are not inferred; 3-5 samples may not cover every cross-product"})
        if args.profiler_dir:
            profiles = profile_samples(args, selected, output, audit, monitor, tokenizer)
        dump_json(output / "profile_metrics.json", profiles)
    # Query limits after steady-state, using actual server prompt usage, not character estimates.
    limits = {}
    for r in records:
        n = r.get("total_input_tokens")
        if n is not None and n not in limits and getattr(args, "backend", "http") != "offline":
            limits[n] = safe_rpc(args, "benchmark_effective_limit", str(n), json.dumps(args.max_tokens))
    for r in records:
        if getattr(args, "backend", "http") == "offline":
            continue
        found = limits.get(r.get("total_input_tokens"), {})
        values = found.get("results", [])
        if values and all(v.get("effective_max_tokens") == values[0].get("effective_max_tokens") for v in values):
            r["effective_max_tokens"] = values[0].get("effective_max_tokens")
            r["effective_max_tokens_source"] = values[0].get("source")
        else:
            r["effective_max_tokens_unavailable_reason"] = found.get("reason", "worker helper unavailable")
    # Atomic replacement retains all observed per-request rows, including failures.
    temp = output / "per_sample_metrics.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(output / "per_sample_metrics.jsonl")
    config.update(steady_start_s=run_start, steady_end_s=run_end, steady_elapsed_s=run_end - run_start,
                  retry_elapsed_s=retry_end - retry_start, worker_memory_before=memory_before,
                  worker_memory_after=memory_after, monitor_errors=monitor.errors,
                  dataset_count=len(samples), worker_telemetry_begin=telemetry_begin,
                  worker_telemetry_scope="host-only input IDs, ready output IDs and scheduler histograms; no per-token synchronization",
                  input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    dump_json(output / "config.json", config)
    dump_json(output / "run_status.json", {"status": "requests_completed", "successes": sum(r["success"] for r in records),
                                           "failures": sum(not r["success"] for r in records)})
    if args.quality_command:
        from benchmark.quality import run_quality_command
        try:
            run_quality_command(args.quality_command, output)
        except Exception as exc:
            dump_json(output / "quality.json", {"status": "failed", "reason": str(exc), "metrics": {}})
    from benchmark.collect_results import collect_run
    summary = collect_run(output)
    try:
        retention = retain_outputs(output, {
            "per_sample_metrics.jsonl": records, "warmup.json": warmups,
            "retry_metrics.json": retries,
            "profile_metrics.json": json.loads((output / "profile_metrics.json").read_text(encoding="utf-8"))},
            mapping, summary["quality"],
            mode=getattr(args, "output_retention", "sampled"),
            sample_count=getattr(args, "retain_samples", 10),
            worst_count=getattr(args, "retain_worst", 10), seed=getattr(args, "retention_seed", 42))
    except Exception as exc:
        retention = {"status": "failed", "reason": str(exc),
                     "note": "retention may be partial; quality and scalar metrics are preserved"}
        dump_json(output / "output_retention.json", retention)
    summary["output_retention"] = retention
    dump_json(output / "summary.json", summary)
    if config.get("server_processor_dir") and summary["server_processor"]["status"] != "measured":
        print("WARNING: SERVER PROCESSOR COST UNAVAILABLE: " + summary["server_processor"]["unavailable_reason"], file=sys.stderr)
    print(f"Results: {output}; inspect acceptance_report.json before claiming bottleneck coverage", flush=True)


if __name__ == "__main__":
    main()
