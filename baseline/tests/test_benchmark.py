"""Targeted, dependency-light tests for metric semantics and collection regressions."""
from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baseline.harness.collect_results import acceptance, collect_run, read_prometheus, throughput
from baseline.harness.metrics import (area_bucket, cuda_event_statistics, dump_json, output_bucket, overlap_duration,
                               stats, union_duration, weighted_stats)
from baseline.harness.prepare_datasets import write_manifest
from baseline.harness.quality import evaluate_baseline, field_quality, edit_distance
from baseline.harness.run_benchmark import (consume_sse, main as benchmark_main, profile_samples, reconcile_worker_records, request_one,
                                     select_representatives, system_tokens_from_chatml, validate_manifest)
from baseline.harness.trace_analysis import analyze_events, collect_traces
from baseline.harness.worker_extension import BenchmarkWorkerExtension, classify_runner_step


def sse(*events):
    return [("data: " + (json.dumps(e) if isinstance(e, dict) else e) + "\n\n").encode() for e in events]


def lines(events):
    return io.BytesIO(b"".join(sse(*events)))


class MetricsTests(unittest.TestCase):
    def test_document_rate_requires_all_page_identities_and_success(self):
        records = [dict(success=True, document_id="doc"), dict(success=False, document_id="doc")]
        self.assertEqual(throughput(records, 1)["documents_per_second"], 0)
        records[1].pop("document_id")
        self.assertIsNone(throughput(records, 1)["documents_per_second"])
        records[1].update(success=True, document_id="doc")
        self.assertEqual(throughput(records, 1)["documents_per_second"], 1)

    def test_distribution_and_null(self):
        self.assertIsNone(stats([])["mean"])
        self.assertEqual(stats([1, 2, None, 3])["count"], 3)
        self.assertEqual(stats([0, 10])["p95"], 9.5)
        self.assertEqual(stats([3])["std"], 0)

    def test_cuda_events_exclude_nested_stages_and_measure_ttft(self):
        def event(index, stage, start, end, parent=None, first=False):
            return dict(id=index, stage=stage, stream=0, start_ms=start, end_ms=end,
                        parent_id=parent, final_prefill_sampling=first)
        events = [event(0, "runner_prefill", 0, 20), event(1, "vision_encoder", 0, 5, 0),
                  event(2, "projector", 2, 3, 1), event(3, "prefill", 5, 15, 0),
                  event(4, "sampling", 15, 16, 0, True), event(5, "decode", 20, 22)]
        value = cuda_event_statistics([{"events": events}])
        self.assertEqual(value["stages"]["vision_encoder"]["exclusive_ms"], 4)
        self.assertEqual(value["stages"]["runner_prefill"]["exclusive_ms"], 4)
        self.assertEqual(value["stages"]["prefill"]["exclusive_ms"], 10)
        self.assertEqual(value["gpu_ttft_ms"], 11)

    def test_stage_host_wall_is_recorded_separately_from_cuda_event(self):
        value = cuda_event_statistics([{"events": [
            dict(id=0, stage="prefill", stream=0, start_ms=0, end_ms=10,
                 parent_id=None, host_call_span_ms=25),
        ]}])
        self.assertEqual(value["stages"]["prefill"]["exclusive_ms"], 10)
        self.assertEqual(value["host_call_spans"]["prefill"]["inclusive_ms"], 25)

    def test_runner_phase_fallback_is_normalized(self):
        value = cuda_event_statistics([{"events": [
            dict(id=0, stage="runner_decode", stream=0, start_ms=0, end_ms=10,
                 parent_id=None),
        ]}])
        self.assertEqual(value["stages"]["decode"]["exclusive_ms"], 10)

    def test_cuda_ttft_does_not_compare_different_stream_anchors(self):
        value = cuda_event_statistics([{"events": [
            dict(id=0, stage="prefill", stream=0, start_ms=0, end_ms=1, parent_id=None),
            dict(id=1, stage="prefill", stream=1, start_ms=0, end_ms=2, parent_id=None),
            dict(id=2, stage="sampling", stream=0, start_ms=2, end_ms=3, parent_id=None, final_prefill_sampling=True)]}])
        self.assertIsNone(value["gpu_ttft_ms"])

    def test_weighted_matches_expanded(self):
        self.assertEqual(weighted_stats({"1": 3, "8": 2}), stats([1, 1, 1, 8, 8]))

    def test_interval_union_and_overlap(self):
        self.assertEqual(union_duration([(0, 10), (2, 3), (8, 12)]), 12)
        self.assertEqual(overlap_duration([(0, 10), (2, 3)], [(5, 20)]), 5)

    def test_bucket_boundaries(self):
        self.assertEqual(area_bucket(1), "0-1 MP")
        self.assertEqual(area_bucket(8), "4-8 MP")
        self.assertEqual(output_bucket(128), "0-128")
        self.assertEqual(output_bucket(129), "129-256")
        self.assertEqual(output_bucket(1024), "513-1024")
        self.assertEqual(output_bucket(None), "unknown")

    def test_prometheus_labels_and_nonfinite(self):
        data = read_prometheus('# TYPE x counter\nvllm:x_total{model_name="a,b"} 3\nx +Inf\ny NaN\n')
        self.assertEqual(data, {('vllm:x_total', '{model_name="a,b"}'): 3.0})

    def test_throughput_does_not_count_directory_files_or_retries(self):
        records = [dict(success=True, finish_reason="stop", num_pages=1, num_images=1,
                        generated_tokens=10, total_input_tokens=20, total_processed_tokens=30),
                   dict(success=False, finish_reason=None)]
        value = throughput(records, 10)
        self.assertEqual(value["images_per_second"], .1)
        self.assertEqual(value["generated_tokens_per_second"], 1)
        self.assertEqual(value["generation_token_coverage"], .5)
        self.assertIsNone(value["documents_per_second"])
        self.assertIsNone(value["prefill_tokens_per_second"])

    def test_missing_token_usage_does_not_become_zero(self):
        value = throughput([dict(success=True, finish_reason="stop", num_pages=1, num_images=1)], 10)
        self.assertIsNone(value["generated_tokens_per_second"])


class StreamingTests(unittest.TestCase):
    def test_one_token_events_have_itl_without_ttft(self):
        values = iter([10., 10.01, 10.03])
        events = [{"id": "r", "choices": [{"delta": {"content": s}, "logprobs": {"content": [{}]}}]} for s in "abc"]
        events += [{"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}, "[DONE]"]
        result = consume_sse(lines(events), clock=lambda: next(values))
        self.assertEqual(result["text"], "abc")
        self.assertEqual(result["first_content_time_s"], 10.)
        self.assertAlmostEqual(result["itl_ms"][0], 10)
        self.assertAlmostEqual(result["itl_ms"][1], 20)

    def test_coalesced_events_are_not_token_itl(self):
        result = consume_sse(lines([{"choices": [{"delta": {"content": "abc"}, "logprobs": {"content": [{}, {}]}}]}, "[DONE]"]))
        self.assertIsNone(result["itl_ms"])
        self.assertEqual(result["events"][0]["token_count"], 2)

    def test_unverified_chunks_are_not_token_itl(self):
        result = consume_sse(lines([{"choices": [{"delta": {"role": "assistant"}}]},
                                    {"choices": [{"delta": {"content": "text"}}]}, "[DONE]"]))
        self.assertEqual(len(result["events"]), 1)
        self.assertIsNone(result["itl_ms"])

    def test_missing_done_is_error(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            consume_sse(lines([{"choices": [{"delta": {"content": "partial"}}]}]))

    def test_stream_error_is_not_success(self):
        with self.assertRaisesRegex(RuntimeError, "bad"):
            consume_sse(lines([{"error": {"message": "bad"}}, "[DONE]"]))


class ManifestTests(unittest.TestCase):
    def test_collision_free_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory in ("a", "b"):
                (root / directory).mkdir()
                (root / directory / "001.png").write_bytes(b"not loaded by manifest")
            output = root / "manifests"
            output.mkdir()
            sidecar = root / "labels.json"
            dump_json(sidecar, {"a/001.png": {"document_type": "invoice"}})
            info = write_manifest("test", root, output, sidecar)
            records = [json.loads(l) for l in Path(info["manifest"]).read_text().splitlines()]
            self.assertEqual(len({r["sample_id"] for r in records}), 2)
            self.assertEqual(records[0]["document_type"], "invoice")
            validate_manifest(records)

    def test_duplicate_and_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            validate_manifest([{"sample_id": "../bad", "images": ["x"]}])


class TraceTests(unittest.TestCase):
    def test_cuda_correlation_exclusive_nested_stages(self):
        events = [
            dict(ph="X", cat="user_annotation", name="benchmark::prefill", ts=0, dur=1000, pid=1, tid=1),
            dict(ph="X", cat="user_annotation", name="benchmark::vision_encoder", ts=10, dur=500, pid=1, tid=1),
            dict(ph="X", cat="cpu_op", name="aten::linear", ts=20, dur=100, pid=1, tid=1, args={"External id": 7}),
            dict(ph="X", cat="user_annotation", name="benchmark::projector", ts=200, dur=200, pid=1, tid=1),
            dict(ph="X", cat="cpu_op", name="aten::mm", ts=210, dur=50, pid=1, tid=1, args={"External id": 8}),
            dict(ph="X", cat="kernel", name="gemm1", ts=100, dur=100, pid=0, tid=7, args={"External id": 7}),
            dict(ph="X", cat="kernel", name="gemm2", ts=300, dur=10, pid=0, tid=7, args={"External id": 8}),
            dict(ph="X", cat="cuda_runtime", name="cudaStreamSynchronize", ts=450, dur=20, pid=1, tid=1),
            dict(ph="X", cat="gpu_memcpy", name="Memcpy HtoD", ts=80, dur=10, pid=0, tid=7, args={"bytes": 4096}),
        ]
        result, ops = analyze_events(events, 20)
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["small_kernel_ratio"], .5)
        self.assertEqual(result["stage_kernel_time_ms"]["vision_encoder"], .1)
        self.assertEqual(result["stage_kernel_time_ms"]["projector"], .01)
        self.assertIsNone(result["stage_kernel_time_ms"]["prefill"])
        self.assertEqual(result["cuda_sync_count"], 1)
        self.assertEqual(result["transfers"]["HtoD"]["bytes"], 4096)
        self.assertEqual(ops[0]["self_cuda_time_total_ms"], .1)
        self.assertAlmostEqual(result["gpu_active_time_ms"], .12)

    def test_cpu_allocations_do_not_become_gpu_peaks(self):
        events = [dict(ph="i", name="[memory]", ts=1, pid=1, tid=1,
                       args={"Device Type": 0, "Total Allocated": 9000000}),
                  dict(ph="i", name="[memory]", ts=2, pid=1, tid=1,
                       args={"Device Type": 1, "Total Allocated": 100, "Total Reserved": 200})]
        result, _ = analyze_events(events)
        self.assertEqual(result["trace_peak_allocated_bytes"], 100)
        self.assertEqual(result["trace_peak_reserved_bytes"], 200)
        self.assertIsNone(result["kernel_count"])

    def test_stage_only_correlation_does_not_claim_operator_hotspots(self):
        events = [dict(ph="X", cat="user_annotation", name="benchmark::decode", ts=0, dur=100, pid=1, tid=1, args={"External id": 1}),
                  dict(ph="X", cat="kernel", name="graph_kernel", ts=1, dur=10, pid=0, tid=0, args={"External id": 1})]
        result, _ = analyze_events(events)
        self.assertEqual(result["correlated_kernel_count"], 1)
        self.assertEqual(result["cuda_operator_attributed_kernel_count"], 0)
        self.assertEqual(result["stage_kernel_time_ms"]["decode"], .01)

    def test_no_attribution_is_null_not_cpu_as_cuda(self):
        result, ops = analyze_events([dict(ph="X", cat="cpu_op", name="aten::mm", ts=0, dur=500, pid=1, tid=1)])
        self.assertIsNone(ops[0]["self_cuda_time_total_ms"])
        self.assertIsNone(result["stage_kernel_time_ms"]["prefill"])
        self.assertIsNone(result["gpu_active_time_ms"])

    def test_graph_replay_kernels_follow_launch_into_stage_range(self):
        # Graph replay: one cudaGraphLaunch without External id launches several kernels.
        events = [
            dict(ph="X", cat="user_annotation", name="benchmark::runner_decode", ts=0, dur=100, pid=1, tid=1),
            dict(ph="X", cat="cuda_runtime", name="cudaGraphLaunch", ts=10, dur=5, pid=1, tid=1, args={"correlation": 7}),
            dict(ph="X", cat="kernel", name="gemm", ts=20, dur=30, pid=0, tid=7, args={"correlation": 7}),
            dict(ph="X", cat="kernel", name="attn", ts=50, dur=20, pid=0, tid=7, args={"correlation": 7}),
            # Launched after the step range closed: must stay unattributed.
            dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=150, dur=5, pid=1, tid=1, args={"correlation": 8}),
            dict(ph="X", cat="kernel", name="copy", ts=160, dur=10, pid=0, tid=7, args={"correlation": 8}),
            # A launch on another thread is not inside this thread's range.
            dict(ph="X", cat="cuda_runtime", name="cudaLaunchKernel", ts=20, dur=5, pid=1, tid=2, args={"correlation": 9}),
            dict(ph="X", cat="kernel", name="other", ts=30, dur=10, pid=0, tid=7, args={"correlation": 9}),
        ]
        result, _ = analyze_events(events)
        self.assertEqual(result["stage_kernel_time_ms"]["decode"], .05)
        self.assertEqual(result["stage_kernel_attribution_counts"],
                         {"external_id_chain": 0, "launch_time_range": 2, "unattributed": 2})
        self.assertEqual(result["correlated_kernel_count"], 0)


class ServerProcessorTests(unittest.TestCase):
    def test_nested_real_methods_preserve_results_and_skip_super_double_count(self):
        import os
        from baseline.harness.server_processor import wrap_method
        class Base:
            def apply(self, image, cached=False):
                return image if cached else self._call_hf_processor(image)
            def _call_hf_processor(self, image):
                return image + 1
        class Child(Base):
            def apply(self, *args, **kwargs):
                return super().apply(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_PROCESSOR_DIR": folder}):
            wrap_method(Base, "apply", "multimodal_apply")
            wrap_method(Base, "_call_hf_processor", "hf_processor")
            wrap_method(Child, "apply", "multimodal_apply")
            self.assertEqual(Child().apply(2), 3)
            self.assertEqual(Child().apply(2, cached=True), 2)
            rows = [json.loads(line) for p in Path(folder).glob("*.jsonl") for line in p.read_text().splitlines()]
            calls = [r for r in rows if r["event"] == "call"]
            self.assertEqual(len(calls), 3)
            hf = next(r for r in calls if r["kind"] == "hf_processor")
            self.assertIn(hf["parent_apply_call_id"], [r["call_id"] for r in calls if r["kind"] == "multimodal_apply"])
            self.assertTrue(all(r["wall_time_ms"] >= 0 and r["thread_cpu_time_ms"] >= 0 for r in calls))
            self.assertTrue(all(r["request_id"] is None for r in calls))

    def test_eight_threads_have_independent_processor_context(self):
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from baseline.harness.server_processor import wrap_method
        barrier = threading.Barrier(8)
        class Processor:
            def apply(self, value):
                barrier.wait(timeout=10)
                return self._call_hf_processor(value)
            def _call_hf_processor(self, value):
                return value * 2
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_PROCESSOR_DIR": folder}):
            wrap_method(Processor, "apply", "multimodal_apply")
            wrap_method(Processor, "_call_hf_processor", "hf_processor")
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(list(pool.map(Processor().apply, range(8))), [i*2 for i in range(8)])
            rows = [json.loads(line) for p in Path(folder).glob("*.jsonl") for line in p.read_text().splitlines()]
            calls = [r for r in rows if r["event"] == "call"]
            self.assertEqual(len(calls), 16)
            parents = [r["parent_apply_call_id"] for r in calls if r["kind"] == "hf_processor"]
            self.assertEqual(len(set(parents)), 8)

    def test_request_context_is_only_attached_when_call_chain_really_encloses_processor(self):
        import os
        from baseline.harness.server_processor import wrap_method
        class Processor:
            def apply(self):
                return 1
        class InputProcessor:
            def process_inputs(self, request_id, prompt):
                return Processor().apply()
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_PROCESSOR_DIR": folder}):
            wrap_method(Processor, "apply", "multimodal_apply")
            wrap_method(InputProcessor, "process_inputs", "input_process")
            self.assertEqual(InputProcessor().process_inputs("req-1", "not logged"), 1)
            Processor().apply()
            rows = [json.loads(line) for p in Path(folder).glob("*.jsonl") for line in p.read_text().splitlines()]
            calls = [r for r in rows if r.get("kind") == "multimodal_apply" and r["event"] == "call"]
            self.assertEqual([r["request_id"] for r in calls], ["req-1", None])

    def test_processor_exception_propagates_and_context_resets(self):
        import os
        from baseline.harness.server_processor import wrap_method
        class Broken:
            def apply(self):
                raise ValueError("private input must not appear in telemetry")
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_PROCESSOR_DIR": folder}):
            wrap_method(Broken, "apply", "multimodal_apply")
            for _ in range(2):
                with self.assertRaises(ValueError):
                    Broken().apply()
            text = next(Path(folder).glob("*.jsonl")).read_text()
            self.assertNotIn("private input", text)
            calls = [json.loads(line) for line in text.splitlines() if json.loads(line)["event"] == "call"]
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(not r["success"] and r["error_type"] == "ValueError" for r in calls))

    def test_bootstrap_in_new_interpreter_hooks_import_and_chains_existing_sitecustomize(self):
        import os
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "vllm" / "multimodal"
            package.mkdir(parents=True)
            (root / "vllm" / "__init__.py").write_text("")
            (package / "__init__.py").write_text("")
            (package / "processing.py").write_text("class BaseMultiModalProcessor:\n    def apply(self, value):\n        return value + 1\n")
            (root / "sitecustomize.py").write_text("import os\nos.environ['BENCHMARK_CHAIN_TEST'] = 'ok'\n")
            logs = root / "logs"
            logs.mkdir()
            repo = Path(__file__).resolve().parents[2]
            env = dict(os.environ, VLLM_BENCHMARK_PROCESSOR_DIR=str(logs),
                PYTHONPATH=os.pathsep.join(map(str, [repo / "baseline/harness/server_bootstrap", repo, root])))
            code = "import os; from vllm.multimodal.processing import BaseMultiModalProcessor; assert BaseMultiModalProcessor().apply(2)==3; assert os.environ['BENCHMARK_CHAIN_TEST']=='ok'"
            result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for p in logs.glob("*.jsonl") for line in p.read_text().splitlines()]
            self.assertTrue(any(r["event"] == "call" for r in calls), result.stderr)

    def test_call_aggregation_is_phase_separated_and_missing_is_not_zero(self):
        from baseline.harness.processor_results import collect_processor_calls
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            def call(start, end, kind):
                return dict(event="call", start_s=start, end_s=end, kind=kind, success=True,
                            wall_time_ms=(end-start)*1000, thread_cpu_time_ms=1)
            events = [call(1, 2, "multimodal_apply"), call(11, 12, "multimodal_apply"),
                      call(11.1, 11.5, "hf_processor"), call(21, 22, "multimodal_apply")]
            (raw / "processor_1.jsonl").write_text("\n".join(json.dumps(r) for r in events))
            config = dict(steady_start_s=10, steady_end_s=20, server_processor_dir=str(raw))
            status = collect_processor_calls(root, config, [], [dict(request_start_s=0, request_end_s=3)])
            self.assertEqual(status["status"], "measured")
            steady = [r for r in status["summaries"] if r["phase"] == "steady"]
            self.assertEqual([r["call_count"] for r in steady], [1, 1])
            self.assertEqual(len((root / "server_processor_calls.jsonl").read_text().splitlines()), 3)
            config.pop("server_processor_dir")
            status = collect_processor_calls(root, config, [])
            self.assertEqual(status["status"], "unavailable")
            self.assertIsNone(status["summaries"][0]["total_wall_time_ms"])


class TraceStorageTests(unittest.TestCase):
    def test_relative_trace_survives_moving_complete_run_without_copy(self):
        import gzip
        import os
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "original"
            raw = original / "profiler_raw"
            raw.mkdir(parents=True)
            run = original / "dataset"
            run.mkdir()
            events = [{"ph": "X", "cat": "kernel", "name": "actual_kernel", "ts": 1, "dur": 10}]
            trace = raw / "worker.json.gz"
            with gzip.open(trace, "wt", encoding="utf-8") as stream:
                json.dump({"traceEvents": events, "other_metadata": "retained"}, stream)
            relative = os.path.relpath(trace, run)
            before = trace.read_bytes()
            moved = Path(folder) / "downloaded"
            shutil.move(str(original), moved)
            run = moved / "dataset"
            profiles = [{"sample_id": "a", "trace_files": [relative]}]
            result = collect_traces(run, profiles)
            self.assertEqual(result[0]["kernel_count"], 1)
            self.assertEqual((moved / "profiler_raw/worker.json.gz").read_bytes(), before)
            reference = json.loads((run / "trace_representative.ref.json").read_text())
            self.assertEqual(reference["trace_file"], relative)
            self.assertTrue((run / reference["trace_file"]).is_file())
            self.assertFalse((run / "trace_representative.json").exists())
            self.assertEqual(len(list((moved / "profiler_raw").iterdir())), 1)
            self.assertEqual(result[0]["trace_file"], relative)

    def test_pruned_traces_keep_statistics_across_recollection(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            profiles = []
            for i, sid in enumerate(("a", "b", "c")):
                dump_json(run / f"raw/{sid}.json", {"traceEvents": [
                    {"ph": "X", "cat": "kernel", "name": "k", "ts": 0, "dur": 10 + i}]})
                profiles.append({"sample_id": sid, "trace_files": [f"raw/{sid}.json"]})
            first = collect_traces(run, profiles, keep_raw=1, seed=42)
            kept = [p["sample_id"] for p in profiles if p["raw_trace_retained"]]
            self.assertEqual(len(kept), 1)
            self.assertEqual(sorted(f.stem for f in (run / "raw").iterdir()), kept)
            reference = json.loads((run / "trace_representative.ref.json").read_text())
            self.assertTrue((run / reference["trace_file"]).is_file())
            first_kernels = (run / "kernel_top50.csv").read_text()
            # A second collect_run pass reads extracted statistics for the deleted traces.
            second = collect_traces(run, profiles, keep_raw=1, seed=42)
            self.assertEqual([r["kernel_count"] for r in second], [r["kernel_count"] for r in first])
            self.assertFalse(any("error" in r for r in second))
            self.assertEqual((run / "kernel_top50.csv").read_text(), first_kernels)

    def test_old_absolute_trace_paths_remain_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            path = run / "old.json"
            dump_json(path, {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": 0, "dur": 2}]})
            result = collect_traces(run, [{"sample_id": "old", "trace_files": [str(path.resolve())]}])
            self.assertEqual(result[0]["kernel_count"], 1)
            self.assertFalse((run / "trace_representative.json").exists())


class WorkerTests(unittest.TestCase):
    def test_runtime_inspection_uses_loaded_objects_without_torch(self):
        class Model:
            def forward(self, value):
                return value
            def named_modules(self):
                return [("", self), ("visual", self)]
        runner = SimpleNamespace(model=Model(), execute_model=lambda scheduler_output: None,
                                 sample_tokens=lambda: None, sampler=SimpleNamespace(forward=lambda value: value))
        worker = BenchmarkWorkerExtension()
        worker.model_runner = runner
        report = worker.benchmark_inspect_runtime()
        self.assertIn("execute_model", report["runner"]["methods"])
        self.assertIn("sample_tokens", report["runner"]["methods"])
        self.assertIn("visual", report["candidate_modules"])
        self.assertIn("forward", report["sampler"]["methods"])

    def test_native_profiler_bound_and_unsupported_interface(self):
        import sys
        worker = BenchmarkWorkerExtension()
        worker.vllm_config = SimpleNamespace(profiler_config=SimpleNamespace(max_iterations=32))
        torch = SimpleNamespace(profiler=SimpleNamespace(profile=type("Profile", (), {})))
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(worker.benchmark_profile_limit()["max_steps"], 32)
            worker.vllm_config.profiler_config.max_iterations = 0
            self.assertFalse(worker.benchmark_profile_limit()["bounded"])
            with self.assertRaises(ValueError):
                worker.benchmark_profile_limit("0")

    def test_legacy_profiler_bound_exports_once_and_restores(self):
        import sys
        exports = []
        schedules = []
        class Profile:
            def __init__(self, **kwargs):
                self.options = kwargs
            def export_chrome_trace(self, path):
                exports.append(path)
                Path(path).write_text('{"traceEvents": []}')
        def schedule(**kwargs):
            schedules.append(kwargs)
            return "bounded_schedule"
        torch = SimpleNamespace(profiler=SimpleNamespace(profile=Profile, schedule=schedule,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda")))
        worker = BenchmarkWorkerExtension()
        original = worker.profiler = Profile()
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {"torch": torch}):
            self.assertTrue(worker.benchmark_profile_limit("8", folder)["bounded"])
            bounded = worker.profiler
            self.assertEqual(schedules[0], dict(wait=0, warmup=0, active=8, repeat=1))
            bounded.options["on_trace_ready"](bounded)
            bounded.export_chrome_trace(str(Path(folder) / "second.json"))
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)
            import gzip
            compressed = Path(folder) / "final.json.gz"
            with patch("gzip.GzipFile", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    bounded.export_chrome_trace(str(compressed))
            self.assertTrue((Path(folder) / "second.json").is_file())
            self.assertFalse(compressed.exists())
            bounded.export_chrome_trace(str(compressed))
            self.assertFalse((Path(folder) / "second.json").exists())
            with gzip.open(compressed, "rt") as stream:
                self.assertEqual(json.load(stream), {"traceEvents": []})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)
            worker.benchmark_profile_limit("8", folder, "true")
            self.assertIs(worker.profiler, original)
            self.assertEqual(len(exports), 1)
            self.assertFalse(worker._benchmark_legacy_profiler_step)
            self.assertTrue(compressed.is_file())

    def test_cuda_event_collection_has_one_final_sync_not_per_token(self):
        import sys
        state = {"clock": 0, "syncs": 0}
        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self, stream):
                self.time = state["clock"]
                state["clock"] += .5
            def elapsed_time(self, other):
                return other.time - self.time
        class Range:
            def __init__(self, *args):
                pass
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass
        def synchronize(*args):
            state["syncs"] += 1
        cuda = SimpleNamespace(current_stream=lambda *a: SimpleNamespace(cuda_stream=0),
                               is_current_stream_capturing=lambda: False, Event=Event, synchronize=synchronize)
        torch = SimpleNamespace(cuda=cuda, profiler=SimpleNamespace(record_function=Range))
        worker = BenchmarkWorkerExtension()
        worker.device = "cuda:0"
        with patch.dict(sys.modules, {"torch": torch}):
            worker.benchmark_event_timing("true")
            for _ in range(3):
                with worker._benchmark_range("decode"):
                    pass
            self.assertEqual(state["syncs"], 0)
            results = worker.benchmark_event_timing("false")
        self.assertEqual(state["syncs"], 1)
        self.assertEqual(len(results["events"]), 3)
        self.assertFalse(worker._benchmark_events_enabled)

    def test_effective_limit_uses_installed_helper(self):
        worker = BenchmarkWorkerExtension()
        worker.model_config = SimpleNamespace(max_model_len=100, get_diff_sampling_param=lambda: {}, override_generation_config={})
        seen = []
        def helper(max_model_len, max_tokens, input_length, default_sampling_params, override_max_tokens=None):
            seen.append(input_length)
            return min(max_model_len - input_length, 7)
        with patch("baseline.harness.worker_extension.importlib.import_module", return_value=SimpleNamespace(get_max_tokens=helper)):
            value = worker.benchmark_effective_limit("80", "null")
        self.assertEqual(value["effective_max_tokens"], 7)
        self.assertEqual(seen, [80])

    def test_native_template_injected_system_is_not_zero(self):
        class Tokenizer:
            all_special_ids = [1, 2]
            def get_vocab(self):
                return {"<|im_start|>": 1, "<|im_end|>": 2}
            def encode(self, text, **kwargs):
                return {"system\n": [3, 4], "user\n": [5, 4], "assistant\n": [6, 4]}[text]
        tokenizer = Tokenizer()
        self.assertEqual(system_tokens_from_chatml([1, 3, 4, 99, 98, 2, 1, 5, 4, 77, 2, 1, 6, 4], tokenizer), 2)
        self.assertEqual(system_tokens_from_chatml([1, 5, 4, 77, 2, 1, 6, 4], tokenizer), 0)
        self.assertIsNone(system_tokens_from_chatml([99, 1, 5, 4, 77, 2], tokenizer))

    def test_profile_ranges_classify_scheduler_and_restore(self):
        import sys
        seen = []
        class Range:
            def __init__(self, name):
                self.name = name
            def __enter__(self):
                seen.append(self.name)
            def __exit__(self, *args):
                pass
        merger = SimpleNamespace(forward=lambda: None)
        visual = SimpleNamespace(forward=lambda: merger.forward())
        model = SimpleNamespace(forward=lambda: visual.forward())
        model.named_modules = lambda: [("", model), ("visual", visual), ("visual.merger", merger)]
        sampler = SimpleNamespace(forward=lambda: None)
        runner = SimpleNamespace(model=model, sampler=sampler, requests={})
        def execute(scheduler):
            model.forward()
            sampler.forward()
        runner.execute_model = execute
        original_model, original_visual = model.forward, visual.forward
        worker = BenchmarkWorkerExtension()
        worker.model_runner = runner
        fake_torch = SimpleNamespace(profiler=SimpleNamespace(record_function=Range))
        request = SimpleNamespace(req_id="r", prompt_token_ids=[1, 2, 3], num_computed_tokens=0)
        scheduler = SimpleNamespace(scheduled_new_reqs=[request], num_scheduled_tokens={"r": 3})
        with patch.dict(sys.modules, {"torch": fake_torch}):
            installed = worker.benchmark_profile_ranges("true")
            runner.execute_model(scheduler)
            worker.benchmark_profile_ranges("false")
        self.assertIn("benchmark::prefill", seen)
        self.assertIn("benchmark::vision_encoder", seen)
        self.assertIn("benchmark::projector", seen)
        self.assertIn("benchmark::sampling", seen)
        self.assertIs(runner.execute_model, execute)
        self.assertIs(model.forward, original_model)
        self.assertIs(visual.forward, original_visual)
        self.assertEqual(installed["installed"]["projector"]["module_path"], "visual.merger")

    def test_telemetry_does_not_force_async_result(self):
        class AsyncResult:
            def get_output(self):
                raise AssertionError("must not synchronize")
        runner = SimpleNamespace(execute_model=lambda *a, **k: AsyncResult())
        worker = BenchmarkWorkerExtension()
        worker.model_runner = runner
        original = runner.execute_model
        worker.benchmark_telemetry("true")
        request = SimpleNamespace(req_id="id", prompt_token_ids=[1, 2], mm_features=[], num_computed_tokens=0)
        scheduled = SimpleNamespace(scheduled_new_reqs=[request], num_scheduled_tokens={"id": 2}, total_num_scheduled_tokens=2)
        runner.execute_model(scheduled)
        value = worker.benchmark_telemetry("false")
        self.assertIs(runner.execute_model, original)
        self.assertEqual(value["requests"]["id"]["prompt_token_ids"], [1, 2])
        self.assertEqual(value["requests"]["id"]["generated_token_ids"], [])
        self.assertEqual(value["steps"]["batch_size_step_counts"], {1: 1})

    def test_reconciliation_uses_request_ids_not_order(self):
        records = [dict(sample_id="a", server_request_id="server-a", total_input_tokens=4, generated_tokens=1)]
        telemetry = {"results": [{"requests": {"server-a-0": {"prompt_token_ids": [1, 2, 2, 3],
                          "multimodal_positions": [{"offset": 1, "length": 2}], "generated_token_ids": [4]}},
                          "steps": {"request_batch_size_histograms": {"server-a-0": {"8": 4}}}}]}
        reconcile_worker_records(records, telemetry)
        self.assertEqual(records[0]["visual_tokens"], 2)
        self.assertEqual(records[0]["actual_scheduler_batch_size"]["mean"], 8)
        self.assertTrue(records[0]["worker_input_length_verified"])

    def test_representatives_unique(self):
        samples = [{"sample_id": str(i)} for i in range(10)]
        records = [dict(sample_id=str(i), success=True, image_megapixels=i, generated_tokens=10-i) for i in range(10)]
        selected = select_representatives(records, samples, 5)
        self.assertEqual(len({r["sample_id"] for r in selected}), 5)


class QualityTests(unittest.TestCase):
    def test_field_quality_exact_and_missing_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pred.md"
            path.write_text('{"name":"Alice","id":"bad"}', encoding="utf-8")
            manifest = [{"sample_id": "a", "ground_truth_fields": {"name": "Alice", "id": "123"}}]
            mapping = [{"sample_id": "a", "success": True, "prediction_path": str(path)}]
            result = field_quality(manifest, mapping)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["metrics"]["field_accuracy"], .5)
            self.assertEqual(result["metrics"]["f1"], .5)
            self.assertEqual(result["counts"]["character_edits"], 3)

    def test_missing_extractor_cannot_pass_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pred.md"
            path.write_text("some markdown", encoding="utf-8")
            result = field_quality([{"sample_id": "a", "ground_truth_fields": {"name": "Alice"}}],
                                   [{"sample_id": "a", "success": True, "prediction_path": str(path)}])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["missing_extractor_samples"], 1)

    def test_failure_is_empty_prediction_not_missing_evaluation(self):
        result = field_quality([{"sample_id": "a", "ground_truth_fields": {"id": "123"}}],
                               [{"sample_id": "a", "success": False}])
        self.assertEqual(result["metrics"]["field_accuracy"], 0)
        self.assertEqual(result["metrics"]["cer"], 1)
        self.assertEqual(result["metrics"]["precision"], 0)


class RequestIntegrationTests(unittest.TestCase):
    @staticmethod
    def args():
        return SimpleNamespace(served_model_name="test", workers=8, max_tokens=None,
                               no_logprobs=False, retry_presence_penalty=1.5,
                               base_url="http://127.0.0.1:26000", request_timeout=10, warmup_request_timeout=10,
                               profile_max_iterations=64)

    def test_request_keeps_usage_and_does_not_send_fixed_cap(self):
        from PIL import Image
        events = [{"id": "chatcmpl-test", "choices": [{"delta": {"content": "hello"}, "logprobs": {"content": [{}, {}]}}]},
                  {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}, "[DONE]"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (16, 8)).save(image)
            sample = {"sample_id": "a", "images": [str(image)]}
            observed = []
            def open_request(request, **kwargs):
                observed.append(json.loads(request.data))
                return lines(events)
            opener = SimpleNamespace(open=open_request)
            with patch("urllib.request.build_opener", return_value=opener):
                record = request_one(sample, self.args(), root)
            self.assertTrue(record["success"], record.get("error"))
            self.assertEqual(record["generated_tokens"], 2)
            self.assertEqual(record["total_processed_tokens"], 14)
            self.assertEqual(record["finish_reason"], "length")
            self.assertTrue(record["length_stopped"])
            self.assertIsNone(record["itl_ms"])
            self.assertNotIn("max_tokens", observed[0])
            self.assertTrue(observed[0]["stream"])
            self.assertIsNone(record["decode_time_ms"])
            self.assertGreaterEqual(record["end_to_end_latency_ms"], record["request_latency_ms"])
            self.assertTrue(Path(record["output_path"]).is_file())

    def test_representative_profiler_stops_before_event_sync(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            args = self.args()
            args.module_map, args.profiler_dir = "{}", raw
            class Response(io.BytesIO):
                status = 200
            def post(url):
                endpoint = url.rsplit("/", 1)[-1]
                calls.append(endpoint)
                if endpoint == "stop_profile":
                    dump_json(raw / "worker.pt.trace.json", {"traceEvents": []})
                return Response(b'{}')
            def rpc(args, method, *values):
                calls.append(method + ":" + str(values[0] if values else ""))
                if method == "benchmark_snapshot":
                    results = [{"peak_allocated_bytes": 100, "peak_reserved_bytes": 200}]
                elif method == "benchmark_profile_limit":
                    results = [{"bounded": True, "max_steps": 64}]
                elif method == "benchmark_event_timing" and values[0] == "false":
                    events = [dict(id=i, stage=s, stream=0, start_ms=i, end_ms=i + .5,
                                   parent_id=None, final_prefill_sampling=s == "sampling")
                              for i, s in enumerate(("vision_encoder", "projector", "prefill", "sampling", "decode"))]
                    results = [{"events": events}]
                else:
                    results = [{}]
                return {"status": "measured", "results": results}
            record = dict(sample_id="a", success=True, generated_tokens=2, end_to_end_latency_ms=100)
            with patch("baseline.harness.run_benchmark.post_json", post), \
                 patch("baseline.harness.run_benchmark.safe_rpc", rpc), \
                 patch("baseline.harness.run_benchmark.request_one", return_value=record):
                profiles = profile_samples(args, [{"sample_id": "a"}], root, {}, SimpleNamespace(phase=None))
            self.assertTrue(profiles[0]["profile_capture_complete"])
            self.assertEqual(profiles[0]["decode_time_ms"], .5)
            self.assertEqual(profiles[0]["tpot_ms"], .25)
            self.assertLess(calls.index("stop_profile"), calls.index("benchmark_event_timing:false"))
            self.assertTrue((root / "profiles/a/cuda_events.json").is_file())
            self.assertEqual(profiles[0]["trace_files"], [str(Path("raw") / "worker.pt.trace.json")])
            self.assertFalse((root / "profiles/a/worker.pt.trace.json").exists())
            reference = json.loads((root / "profiles/a/trace_reference.json").read_text())
            self.assertEqual(reference["trace_files"], profiles[0]["trace_files"])

    def test_empty_response_preserves_finish_and_usage(self):
        from PIL import Image
        events = [{"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, "[DONE]"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (4, 4)).save(image)
            opener = SimpleNamespace(open=lambda *a, **k: lines(events))
            with patch("urllib.request.build_opener", return_value=opener):
                record = request_one({"sample_id": "a", "images": [str(image)]}, self.args(), root)
            self.assertFalse(record["success"])
            self.assertTrue(record["empty_output"])
            self.assertEqual(record["generated_tokens"], 1)
            self.assertEqual(record["finish_reason"], "stop")
            self.assertIsNone(record["output_path"])

    def test_orchestrator_warmup_steady_separation(self):
        from PIL import Image
        class DummyMonitor:
            def __init__(self, *args):
                self.phase = None
                self.errors = []
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        events = [{"id": "test", "choices": [{"delta": {"content": "hello"}, "logprobs": {"content": [{}]}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}, "[DONE]"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (4, 4)).save(image)
            manifest = root / "manifest.jsonl"
            samples = [{"sample_id": f"sample_{i}", "dataset": "midv500", "images": [str(image)]} for i in range(3)]
            manifest.write_text("".join(json.dumps(s) + "\n" for s in samples))
            args = self.args()
            args.manifest, args.output, args.model = manifest, root / "run", "test"
            args.skip_processor_audit, args.trust_remote_code = True, False
            args.gpu, args.server_pid, args.monitor_interval, args.dcgm_url = "1", None, 1, None
            args.warmup, args.representatives, args.profiler_dir = 3, 3, None
            args.retry_repetitive, args.quality_command = False, None
            def fake_rpc(args, method, *values):
                return {"status": "measured", "results": []}
            opener = SimpleNamespace(open=lambda *a, **k: lines(events))
            with patch("baseline.harness.run_benchmark.parse_args", return_value=args), \
                 patch("baseline.harness.run_benchmark.environment", return_value={}), \
                 patch("baseline.harness.run_benchmark.Monitor", DummyMonitor), \
                 patch("baseline.harness.run_benchmark.safe_rpc", fake_rpc), \
                 patch("baseline.harness.run_benchmark.get_url", return_value='{}'), \
                 patch("urllib.request.build_opener", return_value=opener):
                benchmark_main()
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["warmup_latency_ms"]["count"], 3)
            self.assertEqual(summary["tokens"]["generated_tokens"]["count"], 3)
            self.assertEqual(len(list((args.output / "predictions").glob("*.md"))), 3)
            self.assertFalse(json.loads((args.output / "acceptance_report.json").read_text())["all_minimum_criteria_met"])
            self.assertTrue((args.output / "bottleneck_report.md").is_file())


class StartupEvidenceTests(unittest.TestCase):
    def test_async_startup_logs_actual_socket_listen_transition(self):
        import asyncio
        import os
        import socket
        from baseline.harness.http_startup_probe import wrap_async
        class Server:
            started = False
            async def startup(self, sockets):
                sockets[0].listen(2)
                self.started = True
                return "preserved"
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_STARTUP_DIR": folder}),              patch("baseline.harness.http_startup_probe.arm_stacks"), socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            wrap_async(Server, "startup")
            value = asyncio.run(Server().startup([sock]))
            self.assertEqual(value, "preserved")
            events = [json.loads(line) for path in Path(folder).glob("*.jsonl") for line in path.read_text().splitlines()]
            before = next(e for e in events if e["event"] == "enter")
            after = next(e for e in events if e["event"] == "return")
            self.assertFalse(before["state"]["passed_sockets"][0]["accepting"])
            self.assertTrue(after["state"]["passed_sockets"][0]["accepting"])
            self.assertTrue(after["state"]["started"])

    def test_async_startup_exception_is_logged_and_reraised(self):
        import asyncio
        import os
        from baseline.harness.http_startup_probe import wrap_async
        class Server:
            async def startup(self, sockets=None):
                raise ValueError("startup test failure")
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"VLLM_BENCHMARK_STARTUP_DIR": folder}),              patch("baseline.harness.http_startup_probe.arm_stacks"):
            wrap_async(Server, "startup")
            with self.assertRaisesRegex(ValueError, "startup test failure"):
                asyncio.run(Server().startup())
            rows = [json.loads(line) for path in Path(folder).glob("*.jsonl") for line in path.read_text().splitlines()]
            error = next(r for r in rows if r["event"] == "exception")
            self.assertIn("ValueError", error["traceback"])

    def test_proc_tree_excludes_unrelated_processes_and_parses_tcp(self):
        from baseline.harness.network_evidence import process_tree, tcp_entries
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for pid, parent in ((10, 1), (11, 10), (12, 11), (99, 1)):
                directory = root / str(pid)
                directory.mkdir()
                (directory / "stat").write_text(f"{pid} (test process) S {parent} 0 0")
            self.assertEqual(process_tree(10, root), [10, 11, 12])
            tcp = root / "tcp"
            tcp.write_text(chr(10).join(["header", "0: 0100007F:6590 00000000:0000 0A 0:0 0:0 0 0 0 12345", ""]))
            row = tcp_entries(tcp)["entries"][0]
            self.assertEqual(row["port"], 26000)
            self.assertTrue(row["listening"])
            self.assertEqual(row["inode"], "12345")

    def test_evidence_summary_does_not_treat_cancellation_as_startup_error(self):
        from baseline.scripts.summarize_startup_evidence import summarize
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = root / "startup_evidence"
            evidence.mkdir()
            dump_json(root / "server_startup.json", {"status": "failed"})
            dump_json(evidence / "network_final.json", {"client_net_namespace": "net:[1]",
                "processes": [{"pid": 1, "net_namespace": "net:[1]"}, {"pid": 2, "net_namespace": "net:[2]"}],
                "client_namespace_target_entries": [{"port": 26000, "listening": True}]})
            rows = [{"event": "hook_installed"}, {"event": "exception", "error_type": "CancelledError"},
                    {"event": "exception", "error_type": "ValueError"}]
            (evidence / "http_startup_1.jsonl").write_text(chr(10).join(json.dumps(r) for r in rows))
            report = summarize(root)
            self.assertTrue(report["http_hooks_observed"])
            self.assertEqual(report["different_network_namespace_pids"], [2])
            self.assertEqual(len(report["http_task_exceptions"]), 1)
            self.assertEqual(len(report["client_namespace_target_listeners"]), 1)


class ReadinessTests(unittest.TestCase):
    def test_health_then_models_required_for_readiness(self):
        from baseline.harness.readiness import wait_for_server
        calls = []
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        opener = SimpleNamespace(open=lambda url, **kwargs: (calls.append(url) or Response()))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "startup.json"
            result = wait_for_server("http://127.0.0.1:26000", 1, 0, output, opener=opener, alive=lambda: None)
            self.assertEqual(result["status"], "ready")
            self.assertTrue(calls[0].endswith("/health"))
            self.assertTrue(calls[1].endswith("/v1/models"))
            self.assertTrue(output.with_name("readiness_checks.jsonl").is_file())

    def test_http_denial_fails_immediately_with_status(self):
        import urllib.error
        from baseline.harness.readiness import wait_for_server
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        for code in (401, 403, 404):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                def open_url(url, **kwargs):
                    if url.endswith("/health"):
                        return Response()
                    raise urllib.error.HTTPError(url, code, "denied", {}, None)
                output = Path(folder) / "startup.json"
                with self.assertRaises(RuntimeError):
                    wait_for_server("http://127.0.0.1:26000", 1, 0, output,
                        opener=SimpleNamespace(open=open_url), alive=lambda: None,
                        sleep=lambda _: self.fail("must not wait after HTTP denial"))
                report = json.loads(output.read_text())
                self.assertEqual(report["last_probe"]["models"]["status"], code)
                self.assertEqual(report["status"], "failed")
                self.assertIsNone(report["cold_start_latency_s"])

    def test_transport_failure_is_preserved_until_deadline(self):
        import urllib.error
        from baseline.harness.readiness import wait_for_server
        clock = [0]
        def open_url(url, **kwargs):
            raise urllib.error.URLError(ConnectionRefusedError("connection refused"))
        def sleep(seconds):
            clock[0] += seconds
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "startup.json"
            with self.assertRaisesRegex(RuntimeError, "deadline exceeded"):
                wait_for_server("http://localhost:26000", 1, 0, output, timeout=2, interval=1,
                    opener=SimpleNamespace(open=open_url), clock=lambda: clock[0], sleep=sleep, alive=lambda: None)
            data = json.loads(output.read_text())
            self.assertEqual(data["last_probe"]["health"]["error"], "URLError")
            self.assertIn("connection refused", data["last_probe"]["health"]["detail"])
            self.assertEqual(len(output.with_name("readiness_checks.jsonl").read_text().splitlines()), 2)

    def test_application_ready_failure_captures_network_before_cleanup(self):
        import urllib.error
        from baseline.harness.readiness import wait_for_server
        clock = [0]
        def open_url(url, **kwargs):
            raise urllib.error.URLError(ConnectionRefusedError("refused"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / "server.log"
            log.write_text("Application startup complete.")
            with patch("baseline.harness.network_evidence.snapshot", return_value={"target_port": 26000}) as capture:
                with self.assertRaisesRegex(RuntimeError, "diagnostic grace"):
                    wait_for_server("http://127.0.0.1:26000", 1, 0, root / "startup.json", timeout=100,
                        interval=1, opener=SimpleNamespace(open=open_url), clock=lambda: clock[0],
                        sleep=lambda duration: clock.__setitem__(0, clock[0] + duration), alive=lambda: None,
                        diagnostic_dir=root / "evidence", server_log=log, application_ready_grace=3)
                self.assertEqual(clock[0], 3)
                self.assertTrue((root / "evidence/network_final.json").is_file())
                self.assertGreaterEqual(capture.call_count, 2)

    def test_recorded_server_exit_is_not_a_readiness_timeout(self):
        from baseline.harness.readiness import wait_for_server
        def exited():
            raise ProcessLookupError()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "startup.json"
            with self.assertRaisesRegex(RuntimeError, "PID exited"):
                wait_for_server("http://localhost:26000", 1, 0, output, alive=exited)
            self.assertEqual(json.loads(output.read_text())["last_probe"], {})


class RuntimeVerificationTests(unittest.TestCase):
    def test_missing_runtime_files_cannot_verify_anything(self):
        from baseline.scripts.summarize_benchmark_runtime import verify
        with tempfile.TemporaryDirectory() as folder:
            result = verify(folder)
            self.assertFalse(result["all_runtime_checks_verified"])
            self.assertTrue(result["read_errors"])
            self.assertTrue(all(c["status"] == "not_verified" for c in result["checks"]))

    def test_installed_hooks_without_actual_events_do_not_verify_execution(self):
        from baseline.scripts.summarize_benchmark_runtime import verify
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "omnidocbench"
            run.mkdir()
            dump_json(run / "profile_ranges.json", {"results": [{"installed": {s: "fake" for s in ("vision_encoder", "projector", "sampling")}}]})
            result = verify(root)
            checks = {r["check"]: r["status"] for r in result["checks"]}
            self.assertEqual(checks["vision_encoder_hook_installed"], "verified")
            self.assertEqual(checks["vision_encoder_cuda_events_observed"], "not_verified")
            self.assertFalse(result["all_runtime_checks_verified"])

    def test_source_probe_discovers_registration_without_model_initialization(self):
        import ast
        from baseline.scripts.check_benchmark_interfaces import source_evidence
        value = source_evidence(source_evidence)
        self.assertIn("inspect.getsource", value["calls"])
        self.assertIn("signature", value)
        ast.parse(value["source"])


class CollectorIntegrationTests(unittest.TestCase):
    def test_quality_coverage_does_not_imply_baseline_target_met(self):
        quality = {"status": "complete", "required_metrics": ["cer"], "metrics": {"cer": .2}}
        self.assertEqual(evaluate_baseline(quality)["status"], "not_evaluated")
        quality["baseline_thresholds"] = {"cer": {"operator": "<=", "value": .1}}
        self.assertEqual(evaluate_baseline(quality)["status"], "fail")
        quality["metrics"]["cer"] = .05
        self.assertEqual(evaluate_baseline(quality)["status"], "pass")
        quality["status"] = "incomplete"
        self.assertEqual(evaluate_baseline(quality)["status"], "not_evaluated")

    def test_partial_trace_cannot_pass_whole_request_stage_acceptance(self):
        summary = {"memory": {}, "tokens": {}, "startup": {}, "warmup_latency_ms": {"count": 0},
                   "stop_statistics": {"finish_reason_coverage": 0}, "latency_ms": {"count": 0},
                   "throughput": {}, "system": {}}
        trace = {"stage_kernel_time_ms": {s: 1 for s in ("vision_encoder", "prefill", "decode")},
                 "trace_covers_full_request": False}
        def passed(traces, profiles=()):
            return acceptance(summary, [], traces, {}, [], profiles)["criteria"][0]["status"] == "pass"
        self.assertFalse(passed([trace]))
        trace["trace_covers_full_request"] = True
        self.assertTrue(passed([trace]))
        separate = [dict(success=True, **{s + "_time_ms": 1}) for s in ("vision_encoder", "prefill", "decode")]
        self.assertFalse(passed([], separate))
        complete = dict(success=True, **{s + "_time_ms": 1 for s in ("vision_encoder", "prefill", "decode")})
        self.assertTrue(passed([], [complete]))

    def test_official_quality_requires_design_version_and_full_population(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dump_json(root / "config.json", {"steady_start_s": 0, "steady_end_s": 1, "steady_elapsed_s": 1})
            (root / "manifest.jsonl").write_text(json.dumps({"sample_id": "a", "dataset": "omnidocbench"}) + "\n")
            (root / "per_sample_metrics.jsonl").write_text(json.dumps({"sample_id": "a", "success": False,
                "end_to_end_latency_ms": 1, "finish_reason": None}) + "\n")
            quality = {"status": "complete", "benchmark_version": "1.6", "evaluated_samples": 1,
                       "metrics": {"text_edit": .1, "formula_cdm": .9, "table_teds": .8}}
            dump_json(root / "quality.json", quality)
            summary = collect_run(root)
            self.assertEqual(summary["quality"]["status"], "incomplete")
            self.assertFalse(summary["quality"]["version_matches_design"])
            quality.update(benchmark_version="1.5", evaluated_samples=0)
            dump_json(root / "quality.json", quality)
            self.assertFalse(collect_run(root)["quality"]["sample_coverage_complete"])
            quality.update(evaluated_samples=1)
            dump_json(root / "quality.json", quality)
            self.assertEqual(collect_run(root)["quality"]["status"], "complete")

    def test_full_result_package_and_honest_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            dump_json(root / "config.json", {"steady_start_s": 10, "steady_end_s": 20, "steady_elapsed_s": 10})
            sample = {"sample_id": "a", "dataset": "omnidocbench", "images": ["unused"]}
            (root / "manifest.jsonl").write_text(json.dumps(sample) + "\n")
            record = {"sample_id": "a", "success": False, "phase": "steady", "finish_reason": None,
                      "end_to_end_latency_ms": 100, "error": "failure"}
            (root / "per_sample_metrics.jsonl").write_text(json.dumps(record) + "\n")
            dump_json(root / "prediction_mapping.json", [{"sample_id": "a", "success": False}])
            # Stale files do not affect request counts, failures, throughput, or token statistics.
            (root / "predictions").mkdir()
            (root / "predictions" / "stale.md").write_text("old result")
            summary = collect_run(root)
            self.assertEqual(summary["throughput"]["images_per_second"], 0)
            self.assertEqual(summary["stop_statistics"]["error_rate"], 1)
            self.assertIsNone(summary["tokens"]["generated_tokens"]["mean"])
            for name in ("summary.json", "stage_latency.csv", "token_statistics.csv", "padding_statistics.csv",
                         "throughput_statistics.csv", "memory_statistics.csv", "cpu_gpu_transfer.csv", "operator_top50.csv",
                         "kernel_statistics.csv", "server_processor_statistics.csv", "server_processor_calls.jsonl",
                         "server_processor_status.json", "bucket_by_image_area.csv", "bucket_by_visual_tokens.csv",
                         "bucket_by_output_tokens.csv", "metric_coverage.csv", "acceptance_report.json"):
                self.assertTrue((root / name).is_file(), name)
            acceptance = json.loads((root / "acceptance_report.json").read_text())
            self.assertFalse(acceptance["all_minimum_criteria_met"])
            self.assertIn("vision_encoder_profile_kernel_time_ms", acceptance["unavailable_metrics"])
            self.assertEqual(summary["quality"]["missing_metrics"], ["text_edit", "formula_cdm", "table_teds"])
            import csv
            with (root / "metric_coverage.csv").open(encoding="utf-8") as stream:
                inventory = list(csv.DictReader(stream))
            self.assertEqual(len(inventory), len({r["metric"] for r in inventory}))
            self.assertTrue(all(r["status"] == "unavailable" for r in inventory if r["metric"] in ("formula_cdm", "table_teds", "text_edit")))


class SpeculativeDecodingSummaryTest(unittest.TestCase):
    def test_hit_rate_from_counter_deltas(self):
        from baseline.harness.collect_results import speculative_decoding_summary
        row = lambda metric, delta, labels='{engine="0"}': dict(metric=metric, labels=labels, delta=delta)
        rows = [row("vllm:spec_decode_num_drafts_total", 100), row("vllm:spec_decode_num_draft_tokens_total", 1500),
                row("vllm:spec_decode_num_accepted_tokens_total", 600),
                row("vllm:spec_decode_num_accepted_tokens_per_pos_total", 90, '{engine="0",position="0"}'),
                row("vllm:spec_decode_num_accepted_tokens_per_pos_total", 70, '{engine="0",position="1"}'),
                row("vllm:spec_decode_num_drafts_total", None, '{engine="1"}')]  # counter reset: ignored
        s = speculative_decoding_summary(rows)
        self.assertTrue(s["enabled"])
        self.assertEqual(s["acceptance_rate"], .4)
        self.assertEqual(s["mean_acceptance_length"], 7.0)
        self.assertEqual(s["per_position_acceptance_rate"], {0: .9, 1: .7})

    def test_plain_decoding_reports_disabled(self):
        from baseline.harness.collect_results import speculative_decoding_summary
        self.assertFalse(speculative_decoding_summary([])["enabled"])


class RunnerStepClassificationTest(unittest.TestCase):
    @staticmethod
    def step(new=(), cached=(), scheduled=None):
        cached = list(cached)
        return SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=r, prompt_token_ids=[0] * n, num_computed_tokens=c) for r, n, c in new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[r for r, _ in cached], num_computed_tokens=[c for _, c in cached]),
            num_scheduled_tokens=scheduled or {})

    @staticmethod
    def v2_runner(prefill_lens):
        import numpy as np
        return SimpleNamespace(req_states=SimpleNamespace(
            req_id_to_index={rid: i for i, rid in enumerate(prefill_lens)},
            prefill_len=SimpleNamespace(np=np.array(list(prefill_lens.values()), dtype=np.int32))))

    def test_v2_model_runner_decode_is_labeled_decode(self):
        # Regression: V2 runner has no `requests` dict; decode steps were all mixed_or_unknown_step.
        runner = self.v2_runner({"a": 3000})
        self.assertEqual(classify_runner_step(runner, self.step(cached=[("a", 3005)], scheduled={"a": 1}))[0], "decode")
        self.assertEqual(classify_runner_step(runner, self.step(cached=[("a", 2048)], scheduled={"a": 952}))[0], "prefill")

    def test_v1_model_runner_and_new_requests(self):
        runner = SimpleNamespace(requests={"a": SimpleNamespace(num_prompt_tokens=100, prompt_token_ids=None)})
        self.assertEqual(classify_runner_step(runner, self.step(cached=[("a", 120)], scheduled={"a": 1}))[0], "decode")
        self.assertEqual(classify_runner_step(runner, self.step(new=[("b", 50, 0)], scheduled={"b": 50}))[0], "prefill")

    def test_mixed_and_unknown_steps(self):
        runner = self.v2_runner({"a": 100})
        mixed = self.step(new=[("b", 50, 0)], cached=[("a", 120)], scheduled={"a": 1, "b": 50})
        self.assertEqual(classify_runner_step(runner, mixed)[0], "mixed_or_unknown_step")
        unknown = self.step(cached=[("zz", 120)], scheduled={"zz": 1})
        self.assertEqual(classify_runner_step(runner, unknown)[0], "mixed_or_unknown_step")


def fake_checkpoint(root, model_type, **preprocessor):
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "config.json", {"model_type": model_type, "downsample_mode": "4x"} if model_type == "minicpm_ocr"
              else {"model_type": model_type})
    dump_json(root / "preprocessor_config.json", preprocessor)
    return root


class ModelProfileTests(unittest.TestCase):
    def test_routing_preprocessing_and_speculative_permission(self):
        from baseline.harness.model_profiles import check_speculative_config, preprocessor_settings, profile_for
        with tempfile.TemporaryDirectory() as directory:
            minicpm = fake_checkpoint(Path(directory) / "minicpm", "minicpm_ocr", downsample_mode="4x", max_slice_nums=9,
                                      scale_resolution=448, unrelated=1)
            xiaomi = fake_checkpoint(Path(directory) / "xiaomi", "qwen3_5", patch_size=16, merge_size=2)
            profile = profile_for(minicpm)
            self.assertEqual(profile["name"], "MiniCPM-OCR-1.0")
            self.assertTrue(profile["python"].endswith("envs/minicpm_ocr/bin/python"))
            self.assertEqual(preprocessor_settings(minicpm, profile),
                             {"downsample_mode": "4x", "max_slice_nums": 9, "scale_resolution": 448})
            self.assertTrue(profile_for(xiaomi)["python"].endswith("envs/speed_up/bin/python"))
            check_speculative_config(profile, {"method": "dspark", "model": "d"})
            with self.assertRaisesRegex(ValueError, "not enabled"):
                check_speculative_config(profile_for(xiaomi), {"method": "dspark"})
            with self.assertRaisesRegex(ValueError, "no benchmark profile"):
                profile_for(fake_checkpoint(Path(directory) / "other", "llama"))

    def test_parse_args_names_model_and_rejects_unsupported_speculation(self):
        from baseline.harness.run_benchmark import parse_args
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            minicpm = fake_checkpoint(root / "minicpm", "minicpm_ocr")
            xiaomi = fake_checkpoint(root / "xiaomi", "qwen3_5")
            base = ["--backend", "offline", "--gpu", "1", "--manifest", str(root / "m.jsonl")]
            args = parse_args(base + ["--output", str(root / "a"), "--model", str(minicpm),
                                      "--speculative-config", '{"method": "dspark", "model": "d"}'])
            self.assertEqual(args.served_model_name, "MiniCPM-OCR-1.0")
            self.assertEqual(parse_args(base + ["--output", str(root / "b"), "--model", str(xiaomi)]).served_model_name,
                             "Xiaomi-OCR-0")
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                parse_args(base + ["--output", str(root / "c"), "--model", str(xiaomi),
                                   "--speculative-config", '{"method": "dspark"}'])

    def test_minicpm_modules_and_dspark_draft_ranges_install_and_restore(self):
        import sys
        seen = []
        class Range:
            def __init__(self, name):
                self.name = name
            def __enter__(self):
                seen.append(self.name)
            def __exit__(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = fake_checkpoint(Path(directory) / "minicpm", "minicpm_ocr")
            merger = SimpleNamespace(forward=lambda: None)
            vit_merger = SimpleNamespace(forward=lambda: None)
            vpm = SimpleNamespace(forward=lambda: None)
            model = SimpleNamespace(forward=lambda: (vpm.forward(), merger.forward()))
            model.named_modules = lambda: [("", model), ("vpm", vpm), ("vit_merger", vit_merger), ("merger", merger)]
            speculator = SimpleNamespace(propose=lambda: "draft")
            runner = SimpleNamespace(model=model, sampler=None, requests={}, speculator=speculator,
                                     model_config=SimpleNamespace(model=str(checkpoint)))
            original_propose = speculator.propose
            worker = BenchmarkWorkerExtension()
            worker.model_runner = runner
            fake_torch = SimpleNamespace(profiler=SimpleNamespace(record_function=Range))
            with patch.dict(sys.modules, {"torch": fake_torch}):
                result = worker.benchmark_profile_ranges("true")
                model.forward()
                self.assertEqual(speculator.propose(), "draft")
                worker.benchmark_profile_ranges("false")
        self.assertEqual(result["model_profile"], "MiniCPM-OCR-1.0")
        self.assertEqual(result["installed"]["vision_encoder"]["module_path"], "vpm")
        self.assertEqual(result["installed"]["projector"]["module_path"], "merger")
        self.assertIn("draft", result["installed"])
        for stage in ("vision_encoder", "projector", "draft"):
            self.assertIn("benchmark::" + stage, seen)
        self.assertIs(speculator.propose, original_propose)


if __name__ == "__main__":
    unittest.main()
