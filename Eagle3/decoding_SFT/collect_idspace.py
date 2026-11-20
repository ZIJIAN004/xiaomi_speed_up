#!/usr/bin/env python3
r"""Collect IDSpace train.jsonl generations with one local vLLM AsyncLLM.

This reuses the IDSpace prompt, JSONL schema, relative-image resolution, and
native Xiaomi-OCR-0 multimodal processor from idspace/scripts/evaluate_vllm.py
and benchmark/offline_backend.py, but deliberately excludes benchmark
monitoring, profiling, warmup, bottleneck analysis, quality scoring, and HTTP.

PowerShell example:
    python collect_idspace.py `
      --model C:\models\Xiaomi-OCR-0 `
      --input-jsonl C:\data\IDSpace\prepared\train.jsonl `
      --image-root C:\data\IDSpace\prepared\images `
      --output-jsonl C:\data\IDSpace\runs\train_inference.jsonl `
      --max-samples 2
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

PROMPT = """Extract identity information from this ID card.

Instructions:
1. Extract only information visible in the image.
2. Do not infer missing information.
3. Preserve original names and identification numbers.
4. Normalize unambiguous dates to YYYY-MM-DD.
5. Return null for missing or unreadable fields.
6. Return only valid JSON.

Required fields:
{
  \"document_type\": null,
  \"issuing_country\": null,
  \"given_name\": null,
  \"surname\": null,
  \"date_of_birth\": null,
  \"gender\": null,
  \"nationality\": null,
  \"personal_identification_number\": null,
  \"document_number\": null,
  \"issue_date\": null,
  \"expiry_date\": null,
  \"place_of_issue\": null
}"""


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"empty input JSONL: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "image" not in row:
            raise ValueError(f"row {index} must be an object containing image")
    return rows


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    path = Path(row["image"])
    resolved = (path if path.is_absolute() else image_root / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"image not found: {resolved}")
    return resolved


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"animated/multipage image is not supported: {path}")
        image = source.convert("RGB")
        image.load()
    return image


async def collect_stream(outputs):
    text: list[str] = []
    generated_ids: list[int] = []
    prompt_ids = None
    finish_reason = None
    stop_reason = None
    finished = False
    async for output in outputs:
        if output.prompt_token_ids is not None:
            prompt_ids = list(output.prompt_token_ids)
        if len(output.outputs) != 1:
            raise RuntimeError("collector expects exactly one vLLM completion")
        completion = output.outputs[0]
        generated_ids.extend(completion.token_ids)
        text.append(completion.text)
        finish_reason = completion.finish_reason or finish_reason
        stop_reason = completion.stop_reason if completion.stop_reason is not None else stop_reason
        finished = output.finished
    if not finished or finish_reason is None:
        raise RuntimeError("vLLM stream ended without a terminal completion")
    return {
        "response": "".join(text),
        "generated_token_ids": generated_ids,
        "prompt_token_ids": prompt_ids,
        "finish_reason": finish_reason,
        "stop_reason": stop_reason,
        "usage": {
            "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
            "completion_tokens": len(generated_ids),
        },
    }


class LocalVLLMCollector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.engine = None
        self.processor = None
        self.thread.start()
        try:
            self.call(self._initialize(), timeout=args.init_timeout)
        except BaseException:
            self.close()
            raise

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coroutine, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("vLLM operation timed out")

    async def _initialize(self):
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.processor = AutoProcessor.from_pretrained(
            self.args.model,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=self.args.model,
            served_model_name=self.args.served_model_name,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            max_model_len=self.args.max_model_len,
            dtype=self.args.dtype,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            trust_remote_code=self.args.trust_remote_code,
            generation_config="vllm",
            enforce_eager=self.args.enforce_eager,
        ))

    async def _infer(self, image: Image.Image, request_id: str):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        messages = [{"role": "user", "content": [
            {"type": "text", "text": self.args.prompt},
            {"type": "image", "image": image},
        ]}]
        template = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        params = SamplingParams(
            temperature=0,
            top_p=0.8,
            repetition_penalty=1.0,
            max_tokens=self.args.max_tokens,
            output_kind=RequestOutputKind.DELTA,
        )
        outputs = self.engine.generate(
            {"prompt": template, "multi_modal_data": {"image": image}},
            params,
            request_id,
            tokenization_kwargs={"add_special_tokens": False},
        )
        return await asyncio.wait_for(collect_stream(outputs), timeout=self.args.request_timeout)

    def infer(self, image: Image.Image, request_id: str):
        return self.call(self._infer(image, request_id), timeout=self.args.request_timeout + 30)

    def close(self):
        async def shutdown():
            if self.engine is not None:
                result = self.engine.shutdown()
                if hasattr(result, "__await__"):
                    await result
        try:
            if self.engine is not None:
                self.call(shutdown(), timeout=60)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)
            if not self.thread.is_alive():
                self.loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def collect(args: argparse.Namespace) -> None:
    rows = load_rows(args.input_jsonl, args.max_samples)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {args.output_jsonl}")
    with LocalVLLMCollector(args) as collector, args.output_jsonl.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            image_path = resolve_image(row, args.image_root)
            started = time.perf_counter()
            image = load_rgb(image_path)
            try:
                result = collector.infer(image, f"idspace-train-{index:08d}")
            finally:
                image.close()
            record = {
                "sample_id": f"idspace_train_{index:08d}",
                "dataset": "idspace",
                "split": "train",
                "source_image": str(image_path),
                "source_image_relative": row["image"],
                "country": row.get("country"),
                "modality": row.get("modality"),
                "template_id": row.get("template_id"),
                "document_id": row.get("document_id"),
                "target": row.get("target"),
                "prompt": args.prompt,
                "inference": result,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(f"{index + 1}/{len(rows)} {record['sample_id']} elapsed_ms={record['elapsed_ms']:.1f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local Xiaomi-OCR-0 model directory")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="IDSpace train.jsonl")
    parser.add_argument("--image-root", type=Path, required=True, help="local root for relative image paths")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--served-model-name", default="Xiaomi-OCR-0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--init-timeout", type=float, default=1200)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--gpu", default=None, help="CUDA index; sets CUDA_VISIBLE_DEVICES before vLLM initialization")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs; use only if needed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()
    if args.gpu is not None:
        if not args.gpu.isdecimal():
            parser.error("gpu must be a numeric CUDA index")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.max_tokens is not None and args.max_tokens < 1 or args.max_samples < 0 or args.max_model_len < 1:
        parser.error("max-tokens must be positive when provided; max-model-len must be positive; max-samples must be nonnegative")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error(f"local model directory/config.json not found: {args.model}")
    return args


if __name__ == "__main__":
    collect(parse_args())
