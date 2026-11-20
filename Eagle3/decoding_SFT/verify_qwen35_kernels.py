#!/usr/bin/env python3
r"""Verify/install Qwen3.5 linear-attention acceleration in the active env.

This script does not modify environments unless --install is passed. It checks
Python/PyTorch/CUDA/Transformers, imports causal-conv1d and FLA, and can run a
real one-sample Xiaomi-OCR-0 forward to detect Transformers fallback warnings.

Check packages only:
    python verify_qwen35_kernels.py

Resolve packages without installing:
    python verify_qwen35_kernels.py --dry-run

Install packages, then run import checks:
    python verify_qwen35_kernels.py --install

Run one real multimodal forward:
    python verify_qwen35_kernels.py \
      --model /home/mr_jiang/models/xiaomi-ocr-0/model \
      --input-jsonl /home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl \
      --image-root /home/mr_jiang/qkr/runs/idspace_e3/prepared \
      --run-model-test
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="install missing optional kernel packages")
    parser.add_argument("--dry-run", action="store_true", help="ask pip to resolve packages without installing")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--processor", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--run-model-test", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=16384)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def print_environment() -> None:
    print("=== environment ===")
    print("python:", sys.version.replace("\n", " "))
    try:
        import torch
        print("torch:", torch.__version__)
        print("torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as exc:
        print("torch: ERROR", repr(exc))
    for name in ("transformers", "causal-conv1d", "flash-linear-attention", "triton"):
        print(f"{name} distribution:", package_version(name))


def import_check() -> dict[str, bool]:
    print("\n=== imports ===")
    results = {}
    for module_name in ("causal_conv1d", "fla"):
        try:
            module = importlib.import_module(module_name)
            print(module_name, "OK", getattr(module, "__file__", None))
            results[module_name] = True
        except Exception as exc:
            print(module_name, "ERROR", repr(exc))
            results[module_name] = False
    try:
        module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        names = [name for name in dir(module) if "Gated" in name or "Delta" in name]
        print("transformers Qwen3.5 module: OK", names)
        results["transformers_qwen35"] = True
    except Exception as exc:
        print("transformers Qwen3.5 module: ERROR", repr(exc))
        results["transformers_qwen35"] = False
    return results


def pip_action(args: argparse.Namespace) -> int:
    packages = ["causal-conv1d", "flash-linear-attention[cuda]"]
    if not (args.install or args.dry_run):
        return 0
    print("\n=== pip resolution/install ===")
    for package in packages:
        command = [sys.executable, "-m", "pip", "install"]
        if args.dry_run:
            command.append("--dry-run")
        command.append(package)
        print("$", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print("FAILED:", package, "returncode=", result.returncode)
            if args.install:
                return result.returncode
    return 0


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    path = Path(row["image"])
    candidates = [path] if path.is_absolute() else [image_root / path]
    if not path.is_absolute() and path.parts and path.parts[0] == "images":
        candidates.append(image_root.parent / path)
    for candidate in candidates:
        if candidate.resolve().is_file():
            return candidate.resolve()
    raise FileNotFoundError("; ".join(str(item.resolve()) for item in candidates))


def run_model_test(args: argparse.Namespace) -> bool:
    if not args.model or not args.input_jsonl or not args.image_root:
        print("\nmodel test skipped: provide --model, --input-jsonl and --image-root")
        return False
    if not args.model.is_dir() or not args.input_jsonl.is_file():
        print("\nmodel test ERROR: model or input JSONL does not exist")
        return False
    print("\n=== real Xiaomi-OCR-0 forward ===")
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = rows[0]
    image_path = resolve_image(row, args.image_root)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.load()

    processor_path = args.processor or args.model
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
    ).to("cuda:0")
    model.eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Extract the text visible in this image."},
    ]}]
    template = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[template], images=[image], return_tensors="pt", padding=True)
    inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
    if inputs["input_ids"].shape[-1] > args.max_model_len:
        print("model test ERROR: expanded input exceeds max-model-len")
        return False

    captured: list[str] = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True, use_cache=False)
    elapsed = time.perf_counter() - started
    for item in caught:
        text = str(item.message)
        if "causal_conv1d_fn" in text or "chunk_gated_delta_rule" in text:
            captured.append(text)
    print("image:", image_path)
    print("input_ids:", list(inputs["input_ids"].shape))
    print("logits:", list(outputs.logits.shape))
    print("forward_seconds:", round(elapsed, 3))
    if captured:
        print("fallback warnings: PRESENT")
        for text in captured:
            print(" -", text)
        return False
    print("fallback warnings: NOT OBSERVED")
    return True


def main() -> int:
    args = parse_args()
    print_environment()
    pip_code = pip_action(args)
    if pip_code:
        return pip_code
    results = import_check()
    model_ok = True
    if args.run_model_test:
        model_ok = run_model_test(args)
    print("\n=== verdict ===")
    imports_ok = all(results.get(name, False) for name in ("causal_conv1d", "fla", "transformers_qwen35"))
    print("optimized imports:", "PASS" if imports_ok else "FAIL")
    if args.run_model_test:
        print("real model fallback test:", "PASS" if model_ok else "FAIL")
    return 0 if imports_ok and (model_ok or not args.run_model_test) else 2


if __name__ == "__main__":
    raise SystemExit(main())
