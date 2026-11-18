"""Offline adapter tests: no local torch/vLLM installation or GPU required."""
import asyncio
import concurrent.futures
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from benchmark.offline_backend import BASE_URL, OfflineBackend, collect_stream
from benchmark.run_benchmark import parse_args, request_one, reconcile_worker_records
from benchmark.monitor import get_url, rpc


def output(ids, text="", finished=False, reason=None, prompt=(1, 2, 3)):
    return NS(prompt_token_ids=list(prompt), finished=finished,
              outputs=[NS(token_ids=ids, text=text, finish_reason=reason, stop_reason=None)])


async def outputs(*items):
    for item in items:
        yield item


class OfflineStreamTests(unittest.TestCase):
    def test_real_ids_and_token_timestamps(self):
        result = asyncio.run(collect_stream(outputs(output([7], "a"), output([8], "b", True, "stop")),
                                             "rid", clock=iter([1., 1.01]).__next__))
        self.assertEqual(result["text"], "ab")
        self.assertEqual(result["generated_token_ids"], [7, 8])
        self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertAlmostEqual(result["itl_ms"][0], 10)
        self.assertEqual(result["server_request_id"], "rid")

    def test_grouped_tokens_do_not_invent_itl(self):
        result = asyncio.run(collect_stream(outputs(output([7, 8], "ab", True, "length")), "rid"))
        self.assertIsNone(result["itl_ms"])
        self.assertEqual(result["usage"]["completion_tokens"], 2)

    def test_terminal_event_without_ids_does_not_fake_token(self):
        result = asyncio.run(collect_stream(outputs(output([7], "a"), output([], "", True, "stop")), "rid"))
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["usage"]["completion_tokens"], 1)

    def test_unfinished_stream_fails(self):
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            asyncio.run(collect_stream(outputs(output([7], "a")), "rid"))


class FakeBackend(OfflineBackend):
    async def _initialize(self):
        self.engine = NS(shutdown=Mock(), collective_rpc=AsyncMock(return_value=[{"ok": True}]),
                         reset_mm_cache=AsyncMock(), start_profile=AsyncMock(), stop_profile=AsyncMock())
        self.metadata = {"backend": "offline"}


class OfflineBridgeTests(unittest.TestCase):
    def test_rpc_and_controls_never_open_http(self):
        with FakeBackend(NS()) as backend, patch("urllib.request.build_opener", side_effect=AssertionError("HTTP forbidden")):
            self.assertEqual(rpc(BASE_URL, "benchmark_snapshot", ["false"]), [{"ok": True}])
            self.assertEqual(json.loads(get_url(BASE_URL + "/server_info")), {"backend": "offline"})
            from benchmark.run_benchmark import post_json
            with post_json(BASE_URL + "/start_profile") as response:
                self.assertEqual(response.status, 200)
            with post_json(BASE_URL + "/reset_mm_cache"):
                pass
            backend.engine.collective_rpc.assert_awaited_once_with("benchmark_snapshot", timeout=600, args=("false",))
            backend.engine.start_profile.assert_awaited_once()
        self.assertFalse(backend.thread.is_alive())
        backend.engine.shutdown.assert_called_once()

    def test_eight_requests_share_one_live_loop(self):
        async def work(i):
            await asyncio.sleep(0.001)
            return i, id(asyncio.get_running_loop())
        with FakeBackend(NS()) as backend:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda i: backend.call(work(i)), range(16)))
        self.assertEqual(sorted(i for i, _ in results), list(range(16)))
        self.assertEqual(len({loop for _, loop in results}), 1)

    def test_native_prometheus_without_http(self):
        from types import ModuleType
        import sys
        module = ModuleType("prometheus_client")
        module.REGISTRY = object()
        module.CollectorRegistry = Mock()
        module.multiprocess = NS(MultiProcessCollector=Mock())
        module.generate_latest = Mock(return_value=b"vllm:test_total 2\n")
        with FakeBackend(NS()) as backend, patch.dict(sys.modules, {"prometheus_client": module}), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_url(BASE_URL + "/metrics"), "vllm:test_total 2\n")
            module.generate_latest.assert_called_once_with(module.REGISTRY)

    def test_parser_defaults_offline_and_keeps_output_unbounded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--manifest", "input.jsonl", "--output", str(Path(directory)/"new"), "--model", "model"])
        self.assertEqual(args.backend, "offline")
        self.assertEqual(args.base_url, BASE_URL)
        self.assertEqual(args.workers, 8)
        self.assertIsNone(args.max_tokens)

    def test_offline_request_writes_predictions_and_keeps_metrics(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (20, 20)).save(image)
            args = NS(backend="offline", served_model_name="model", workers=8, max_tokens=None)
            result = {"text": "hello", "engine_submit_s": time.perf_counter(), "chat_template_time_ms": 1,
                      "effective_max_tokens": 100, "generated_token_ids": [7, 8], "prompt_token_ids": [1, 2, 3],
                      "finish_reason": "stop", "server_request_id": "benchmark-steady-a-0", "events": [],
                      "chunk_interval_ms": [], "itl_ms": None, "itl_source": None,
                      "itl_unavailable_reason": "grouped", "first_content_time_s": None,
                      "last_content_time_s": None, "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
            backend = NS(infer=Mock(return_value=result))
            with patch("benchmark.offline_backend._ACTIVE", backend), patch("urllib.request.build_opener", side_effect=AssertionError("HTTP forbidden")):
                record = request_one({"sample_id": "a", "images": [str(image)]}, args, root)
            self.assertTrue(record["success"], record["error"])
            self.assertEqual(record["generated_tokens"], 2)
            self.assertEqual(record["total_input_tokens"], 3)
            self.assertEqual(record["effective_max_tokens"], 100)
            self.assertIsNone(record["serialization_time_ms"])
            self.assertNotIn("http_start_s", record)
            self.assertEqual(Path(record["output_path"]).read_text(), "hello")

    def test_output_classification_survives_missing_worker_ids(self):
        record = {"success": False, "generated_tokens": 2, "offline_generated_token_ids": [7, 8], "finish_reason": "stop"}
        tokenizer = NS(all_special_ids=[8], eos_token_id=8)
        reconcile_worker_records([record], {"results": []}, tokenizer)
        self.assertEqual(record["generated_visible_tokens"], 1)
        self.assertTrue(record["stop_reason_is_eos"])
        self.assertEqual(record["generated_token_classification_source"], "actual_offline_delta_token_ids")


class OfflineInferenceTests(unittest.TestCase):
    def test_sampling_template_limits_and_abort(self):
        from types import ModuleType
        import sys
        vllm = ModuleType("vllm")
        vllm.SamplingParams = Mock(side_effect=lambda **kw: NS(**kw))
        sampling = ModuleType("vllm.sampling_params")
        sampling.RequestOutputKind = NS(DELTA="delta")
        backend = object.__new__(OfflineBackend)
        backend.processor = NS(apply_chat_template=Mock(return_value="native template"))
        backend.engine = NS(generate=Mock(return_value=outputs(output([7], "a", True, "stop"))), abort=AsyncMock())
        args = NS(max_tokens=None, max_model_len=100, retry_presence_penalty=1.5, no_logprobs=True, request_timeout=1)
        with patch.dict(sys.modules, {"vllm": vllm, "vllm.sampling_params": sampling}):
            result = asyncio.run(backend._infer("image", args, "rid", 0))
            self.assertEqual(result["effective_max_tokens"], 97)
            params = vllm.SamplingParams.call_args.kwargs
            self.assertIsNone(params["max_tokens"])
            self.assertEqual(params["output_kind"], "delta")
            self.assertEqual(params["temperature"], 0)
            self.assertEqual(backend.engine.generate.call_args.kwargs, {"tokenization_kwargs": {"add_special_tokens": False}})
            self.assertFalse(backend.processor.apply_chat_template.call_args.kwargs["tokenize"])
            self.assertFalse(backend.processor.apply_chat_template.call_args.kwargs["enable_thinking"])
            async def failure():
                raise ValueError("generation failed")
                yield
            backend.engine.generate.return_value = failure()
            with self.assertRaisesRegex(ValueError, "generation failed"):
                asyncio.run(backend._infer("image", args, "failed", 0))
            backend.engine.abort.assert_awaited_once_with("failed")

    def test_three_datasets_load_one_engine(self):
        from benchmark.run_offline import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            images = root / "images"
            images.mkdir()
            (images / "a.png").write_bytes(b"manifest discovery only")
            calls = []
            def run(args):
                calls.append(args)
                args.output.mkdir()
                (args.output / "acceptance_report.json").write_text("{}")
            instance = NS(metadata={"backend": "offline"}, rpc=Mock(return_value=[{"peak_allocated_bytes": 100}]))
            context = Mock(__enter__=Mock(return_value=instance), __exit__=Mock(return_value=False))
            argv = ["offline", str(images), str(images), str(images), str(model), str(root / "results")]
            with patch("sys.argv", argv), patch.dict("os.environ", {}, clear=True), patch("benchmark.offline_backend.OfflineBackend", return_value=context) as constructor, patch("benchmark.run_offline.run_dataset", side_effect=run):
                main()
            constructor.assert_called_once()
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(a.workers == 8 and a.backend == "offline" and a.max_model_len == 16384 for a in calls))
            self.assertEqual(len({str(a.profiler_dir) for a in calls}), 1)
            startup = json.loads((root / "results/server_startup.json").read_text())
            self.assertEqual(startup["backend"], "offline")


class OfflineEndToEndTests(unittest.TestCase):
    def test_full_collection_without_any_http(self):
        from PIL import Image
        from benchmark.run_benchmark import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "a.png"
            Image.new("RGB", (10, 10)).save(image)
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps({"sample_id": "s"+str(i), "dataset": "omnidocbench", "images": [str(image)]}) for i in range(3)))
            args = parse_args(["--manifest", str(manifest), "--output", str(root / "run"), "--model", "model", "--warmup", "3", "--skip-processor-audit"])
            tokenizer = NS(all_special_ids=[], eos_token_id=8, encode=lambda text, **kw: [7])
            def infer(image, args, rid, attempt):
                now = time.perf_counter()
                return {"text": "hello", "engine_submit_s": now, "chat_template_time_ms": 0.1,
                        "effective_max_tokens": 100, "generated_token_ids": [7], "prompt_token_ids": [1,2],
                        "finish_reason": "stop", "server_request_id": rid,
                        "events": [{"time_s": now, "token_count": 1}], "chunk_interval_ms": [],
                        "itl_ms": [], "itl_source": "actual", "itl_unavailable_reason": None,
                        "first_content_time_s": now, "last_content_time_s": now,
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1}}
            backend = NS(metadata={"backend": "offline"}, processor=NS(tokenizer=tokenizer), infer=infer,
                         rpc=Mock(return_value=[]), get=lambda endpoint: "" if endpoint == "metrics" else "{}")
            monitor = NS(errors=[], phase="")
            context = Mock(__enter__=Mock(return_value=monitor), __exit__=Mock(return_value=False))
            with patch("benchmark.offline_backend._ACTIVE", backend), patch("benchmark.run_benchmark.Monitor", return_value=context), patch("urllib.request.build_opener", side_effect=AssertionError("HTTP forbidden")):
                main(args)
            records = [json.loads(line) for line in (root / "run/per_sample_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["success"] and r["generated_tokens"] == 1 for r in records))
            self.assertTrue((root / "run/metric_coverage.csv").is_file())
            self.assertTrue((root / "run/metric_migration.json").is_file())
            summary = json.loads((root / "run/summary.json").read_text())
            self.assertGreater(summary["throughput"]["generated_tokens_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
