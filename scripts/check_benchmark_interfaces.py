#!/usr/bin/env python3
"""Read-only installed-runtime diagnostics; no downloads, model startup or package changes."""
from __future__ import annotations

import argparse
import ast
import textwrap
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


def source_evidence(value):
    """Installed source evidence only; never instantiate a model."""
    try:
        source = textwrap.dedent(inspect.getsource(value))
        tree = ast.parse(source)
        calls = sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)})
        attributes = sorted({node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)})
        return {"source_file": inspect.getsourcefile(value), "first_line": inspect.getsourcelines(value)[1],
                "signature": str(inspect.signature(value)), "calls": calls, "attributes": attributes,
                "source": source[:30000], "source_truncated": len(source) > 30000}
    except Exception as exc:
        return {"error_type": type(exc).__name__, "detail": str(exc)}


def deep_probe():
    result = {"scope": "installed source and schema evidence; no runtime invocation proved", "modules": {}}
    names = (
        "vllm.model_executor.models.qwen3_5", "vllm.model_executor.models.qwen3_vl",
        "vllm.multimodal.processing.processor", "vllm.v1.engine.input_processor",
        "vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu_worker",
        "vllm.v1.core.sched.output", "vllm.v1.outputs",
    )
    for name in names:
        print("Inspecting: " + name, file=sys.stderr, flush=True)
        entry = {}
        try:
            module = importlib.import_module(name)
            entry["source_file"] = getattr(module, "__file__", None)
            classes = {}
            for alias, cls in vars(module).items():
                if not inspect.isclass(cls):
                    continue
                processor = any(base.__name__ == "BaseMultiModalProcessor" for base in cls.__mro__)
                own = cls.__module__ == name
                if not processor and not own:
                    continue
                methods = {method: source_evidence(getattr(cls, method)) for method in
                    ("apply", "_call_hf_processor", "process_inputs", "_process_inputs", "execute_model", "sample_tokens", "profile")
                    if callable(getattr(cls, method, None))}
                fields = [f.name for f in dataclasses.fields(cls)] if dataclasses.is_dataclass(cls) else list(getattr(cls, "__annotations__", {}))
                if methods or fields or processor:
                    classes[alias] = {"defined_in": cls.__module__, "mro": [c.__name__ for c in cls.__mro__],
                                      "methods": methods, "fields": fields}
            entry["classes"] = classes
            if name.endswith("qwen3_5"):
                tree = ast.parse(inspect.getsource(module))
                entry["processor_registrations"] = [ast.unparse(decorator) for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef) for decorator in node.decorator_list
                    if "register_processor" in ast.unparse(decorator)]
            entry["status"] = "inspected"
        except Exception as exc:
            entry.update(status="unavailable", error_type=type(exc).__name__, detail=str(exc))
        result["modules"][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="include installed method source, processor registration and scheduler/output schemas; no model load")
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
        ("vllm.v1.engine.input_processor", "InputProcessor", ("process_inputs",)),
        ("vllm.config", "ProfilerConfig", ()),
        ("vllm.v1.worker.gpu_worker", "Worker", ("profile", "execute_model")),
    ]
    print("Inspecting installed interfaces (no model weights loaded)", file=sys.stderr, flush=True)
    report["interfaces"] = [inspect_target(*target) for target in targets]
    if args.deep:
        report["deep_inspection"] = deep_probe()
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
