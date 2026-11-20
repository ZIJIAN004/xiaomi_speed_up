#!/usr/bin/env python3
"""Collect direct Xiaomi-OCR-0 inference records from IDSpace train JSONL.

This is a data-collection runner, not a bottleneck benchmark. It performs one
native multimodal generation per selected IDSpace sample and writes one JSONL
record per sample, including the source image, prompt, target annotation, raw
model response, and actual token IDs when the backend exposes them.

Example (direct Transformers backend):
    python collect_idspace.py \
        --model C:\\models\\Xiaomi-OCR-0 \
        --input-jsonl C:\\data\\idspace\\prepared\\train.jsonl \
        --image-root C:\\data\\idspace\\prepared\\images \
        --output-jsonl C:\\data\\idspace\\runs\\train_inference.jsonl

Example (vLLM OpenAI-compatible server):
    python collect_idspace.py \
        --backend openai \
        --base-url http://127.0.0.1:8000/v1 \
        --model SeerRay-Lab/Xiaomi-OCR-0 \
        --input-jsonl C:\\data\\idspace\\prepared\\train.jsonl \
        --image-root C:\\data\\idspace\\prepared\\images \
        --output-jsonl C:\\data\\idspace\\runs\\train_inference.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
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


def encode_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def run_openai(args: argparse.Namespace, image: Image.Image, prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_png_data_url(image)}},
        ]}],
        temperature=0,
        top_p=0.8,
        max_tokens=args.max_tokens,
        stream=False,
    )
    choice = response.choices[0]
    usage = response.usage.model_dump() if response.usage is not None else None
    return {
        "response": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "generated_token_ids": None,
        "prompt_token_ids": None,
        "usage": usage,
        "backend": "openai-compatible",
    }


def build_transformers_runner(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.eval()

    def generate(image: Image.Image, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        template = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=[template],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_length:].detach().cpu().tolist()
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return {
            "response": text,
            "finish_reason": "stop" if len(generated_ids) < args.max_tokens else "length",
            "generated_token_ids": generated_ids,
            "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
            "usage": {"prompt_tokens": prompt_length, "completion_tokens": len(generated_ids)},
            "backend": "transformers",
        }

    return generate


def collect(args: argparse.Namespace) -> None:
    rows = load_rows(args.input_jsonl, args.max_samples)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {args.output_jsonl}")

    generate = None if args.backend == "openai" else build_transformers_runner(args)
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            image_path = resolve_image(row, args.image_root)
            started = time.perf_counter()
            image = load_rgb(image_path)
            try:
                result = (run_openai(args, image, args.prompt) if args.backend == "openai"
                          else generate(image, args.prompt))
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
    parser.add_argument("--model", required=True, help="Local model path or served model name")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="IDSpace train.jsonl")
    parser.add_argument("--image-root", type=Path, required=True, help="Root for relative image paths")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_samples < 0:
        parser.error("max-tokens must be positive and max-samples must be nonnegative")
    return args


if __name__ == "__main__":
    collect(parse_args())
