#!/usr/bin/env python3
"""Read-only installed-runtime diagnostics; no downloads, model startup or package changes."""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path


def inspect_target(module_name, class_name, methods):
    result = {"module": module_name, "class": class_name}
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        result["source_file"] = inspect.getsourcefile(cls)
        result["methods"] = {}
        for name in methods:
            method = getattr(cls, name, None)
            if method is not None:
                result["methods"][name] = {"signature": str(inspect.signature(method)),
                    "defined_in": getattr(method, "__module__", None),
                    "async": inspect.iscoroutinefunction(method)}
        if dataclasses.is_dataclass(cls):
            result["dataclass_fields"] = [f.name for f in dataclasses.fields(cls)]
        result["status"] = "available"
    except Exception as exc:
        result.update(status="unavailable", error_type=type(exc).__name__, detail=str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new JSON report path; otherwise stdout")
    parser.add_argument("--model-path", type=Path, help="existing local model directory; read JSON metadata only")
    args = parser.parse_args()
    if args.model_path and not args.model_path.is_dir():
        parser.error("model-path must be an existing local directory")
    if args.output and args.output.exists():
        parser.error("output already exists; refusing overwrite")
    report = {"python": sys.version, "platform": platform.platform(), "packages": {},
              "scope": "imports/introspection only; no processor/model instantiated or inference validated"}
    for name in ("vllm", "torch", "transformers", "tokenizers", "numpy", "Pillow"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda
    except Exception as exc:
        report["torch_import_error"] = str(exc)
    targets = [
        ("vllm.multimodal.processing", "BaseMultiModalProcessor", ("apply", "_call_hf_processor", "_cached_apply_hf_processor")),
        ("vllm.multimodal.processing.processor", "BaseMultiModalProcessor", ("apply", "_call_hf_processor", "_cached_apply_hf_processor")),
        ("vllm.model_executor.models.qwen3_5", "Qwen3_5MultiModalProcessor", ("apply", "_call_hf_processor")),
        ("vllm.v1.engine.input_processor", "InputProcessor", ("process_inputs",)),
        ("vllm.inputs.preprocess", "InputPreprocessor", ("preprocess",)),
        ("vllm.config", "ProfilerConfig", ()),
        ("vllm.v1.worker.gpu_worker", "Worker", ("profile", "execute_model")),
    ]
    report["interfaces"] = [inspect_target(*target) for target in targets]
    if args.model_path:
        report["local_model_metadata"] = {}
        for name in ("config.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
            path = args.model_path / name
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    keys = ("architectures", "model_type", "auto_map", "processor_class", "image_processor_type",
                            "patch_size", "merge_size", "do_resize", "do_normalize", "image_mean", "image_std", "size")
                    report["local_model_metadata"][name] = {k: data[k] for k in keys if k in data}
                except Exception as exc:
                    report["local_model_metadata"][name] = {"error": str(exc)}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=False)
        report["gpu"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["gpu"] = {"error": str(exc)}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print("Saved:", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
