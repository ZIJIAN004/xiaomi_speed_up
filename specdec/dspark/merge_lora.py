#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base MiniCPM-OCR checkpoint, tensor by tensor.

W' = W + (lora_alpha / r) * B @ A, computed in fp32 and stored in W's dtype. The
merge works on the safetensors files directly: every key name, shard file, dtype
and config.json stays the base model's, so the result loads wherever the base
does (HF for DSpark training, vLLM for serving). Non-weight files are copied from
the base. The adapter's tokenizer/processor files are compared with the base's
and reported in merge_info.json.

Plain LoRA only (no DoRA, rsLoRA, fan_in_fan_out, bias, modules_to_save, rank or
alpha patterns); anything else is refused.

``--verify-samples N`` (needs peft, run in the env that trained the LoRA) then
compares, on N validation KIE samples rendered as in training, the teacher-forced
next-token predictions at the answer positions of: the base model, base + adapter
through PEFT (unmerged), and the merged checkpoint. It fails if merged and PEFT
argmax agree on fewer than ``--min-agreement`` of the positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROCESSOR_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
                   "preprocessor_config.json")
_KEY = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<which>[AB])\.weight$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="base model directory (the LoRA's base)")
    p.add_argument("--adapter", type=Path, required=True, help="PEFT adapter directory")
    p.add_argument("--output", type=Path, required=True, help="merged model directory (must not exist)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --output")
    p.add_argument("--skip-merge", action="store_true", help="only run --verify-samples on an existing --output")
    p.add_argument("--verify-samples", type=int, default=0, help="KIE validation samples to compare (needs peft)")
    p.add_argument("--min-agreement", type=float, default=0.995)
    p.add_argument("--data-dir", type=Path, default=None, help="LoRA data run dir (validation.jsonl, images/)")
    p.add_argument("--layout-jsonl", type=Path, default=None, help="default: <data-dir>/layout.jsonl")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_scaling(config: dict) -> float:
    unsupported = {
        "peft_type": config.get("peft_type") != "LORA",
        "use_dora": bool(config.get("use_dora")),
        "use_rslora": bool(config.get("use_rslora")),
        "fan_in_fan_out": bool(config.get("fan_in_fan_out")),
        "bias": config.get("bias", "none") != "none",
        "lora_bias": bool(config.get("lora_bias")),
        "modules_to_save": bool(config.get("modules_to_save")),
        "rank_pattern": bool(config.get("rank_pattern")),
        "alpha_pattern": bool(config.get("alpha_pattern")),
    }
    bad = [k for k, v in unsupported.items() if v]
    if bad:
        raise ValueError(f"adapter uses features this merge does not implement: {bad}")
    return float(config["lora_alpha"]) / float(config["r"])


def read_adapter(adapter_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{base weight key: (A [r, in], B [out, r])}."""
    from safetensors.torch import load_file

    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        match = _KEY.match(key)
        if not match:
            raise ValueError(f"unexpected adapter tensor {key}")
        pairs.setdefault(match["module"] + ".weight", {})[match["which"]] = value
    incomplete = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"adapter modules without both lora_A and lora_B: {incomplete[:5]}")
    return {k: (v["A"], v["B"]) for k, v in pairs.items()}


def merge(args) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = lora_scaling(config)
    pairs = read_adapter(args.adapter)
    shards = sorted(args.base.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {args.base}")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists (pass --overwrite to replace it)")
    tmp = args.output.with_name(args.output.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    merged, rel_norms = set(), []
    for shard in shards:
        with safe_open(str(shard), "pt") as source:
            metadata = source.metadata()
            tensors = {key: source.get_tensor(key) for key in source.keys()}
        for key in tensors:
            if key not in pairs:
                continue
            weight = tensors[key]
            lora_a, lora_b = pairs[key]
            delta = (lora_b.double() @ lora_a.double()) * scaling
            if delta.shape != weight.shape:
                raise ValueError(f"{key}: delta {tuple(delta.shape)} != weight {tuple(weight.shape)}")
            new = (weight.double() + delta).to(weight.dtype)
            rel_norms.append(float(delta.norm() / weight.double().norm().clamp_min(1e-12)))
            tensors[key] = new.contiguous()
            merged.add(key)
        save_file(tensors, str(tmp / shard.name), metadata=metadata)
        print(f"wrote {shard.name}: {len(tensors)} tensors", flush=True)
    unmatched = sorted(set(pairs) - merged)
    if unmatched:
        shutil.rmtree(tmp)
        raise KeyError(f"adapter modules with no base weight: {unmatched[:5]}")

    for item in args.base.iterdir():
        if item.name.endswith(".safetensors") or item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, tmp / item.name)
        else:
            shutil.copy2(item, tmp / item.name)

    processor_files = {}
    for name in PROCESSOR_FILES:
        base_file, adapter_file = args.base / name, args.adapter / name
        if adapter_file.is_file():
            processor_files[name] = ("identical" if base_file.is_file() and sha256(base_file) == sha256(adapter_file)
                                     else "differs (base copy kept)")
    info = {
        "base": str(args.base), "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "adapter_base_model_name_or_path": config.get("base_model_name_or_path"),
        "r": config["r"], "lora_alpha": config["lora_alpha"], "scaling": scaling,
        "merged_modules": len(merged), "delta_rel_norm_max": max(rel_norms),
        "delta_rel_norm_mean": sum(rel_norms) / len(rel_norms),
        "adapter_processor_files_vs_base": processor_files,
    }
    (tmp / "merge_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if args.output.exists():
        shutil.rmtree(args.output)
    tmp.rename(args.output)
    print(json.dumps(info, indent=2), flush=True)
    differing = [k for k, v in processor_files.items() if v != "identical"]
    if differing:
        print(f"WARNING: adapter files differ from the base: {differing}; the merged model uses the base copies",
              flush=True)
    return info


@torch.no_grad()
def answer_predictions(model, samples, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(argmax, float logits, ground-truth next token) at every loss-mask position, concatenated."""
    preds, logits_all, gold = [], [], []
    for raw in samples:
        inputs = {k: v.to(device) for k, v in raw["inputs"].items()}
        pos = raw["loss_mask"][0].nonzero().squeeze(-1)
        logits = model(**inputs, use_cache=False, return_dict=True).logits[0, pos.to(device)].float()
        preds.append(logits.argmax(-1).cpu())
        logits_all.append(logits.cpu())
        gold.append(raw["inputs"]["input_ids"][0, pos + 1])
    return torch.cat(preds), torch.cat(logits_all), torch.cat(gold)


def verify(args) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from idspace_minicpm_dataset import make_datasets_factory

    if args.data_dir is None:
        raise ValueError("--verify-samples needs --data-dir")
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.base)
    data_args = SimpleNamespace(task="kie", layout_jsonl=args.layout_jsonl or args.data_dir / "layout.jsonl",
                                max_pixels=262144, max_slice_nums=1, prompt_file=None, text_first=False,
                                image_root=args.data_dir)
    dataset = make_datasets_factory(data_args, processor)(args.data_dir / "validation.jsonl", args.verify_samples)
    samples = [dataset[i] for i in range(len(dataset))]

    def load(path):
        return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()

    model = load(args.output)
    merged_pred, merged_logits, gold = answer_predictions(model, samples, device)
    del model
    model = PeftModel.from_pretrained(load(args.base), str(args.adapter)).eval()
    peft_pred, peft_logits, _ = answer_predictions(model, samples, device)
    with model.disable_adapter():
        base_pred, _, _ = answer_predictions(model, samples, device)
    del model

    report = {
        "samples": len(samples), "positions": int(gold.numel()),
        "merged_vs_peft_argmax_agreement": float((merged_pred == peft_pred).float().mean()),
        "merged_vs_peft_logit_max_abs": float((merged_logits - peft_logits).abs().max()),
        "merged_vs_peft_logit_mean_abs": float((merged_logits - peft_logits).abs().mean()),
        "teacher_forced_acc_base": float((base_pred == gold).float().mean()),
        "teacher_forced_acc_peft": float((peft_pred == gold).float().mean()),
        "teacher_forced_acc_merged": float((merged_pred == gold).float().mean()),
        "min_agreement": args.min_agreement,
    }
    report["passed"] = report["merged_vs_peft_argmax_agreement"] >= args.min_agreement
    (args.output / "merge_verify.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    args = parse_args()
    if not args.skip_merge:
        merge(args)
    elif not args.output.is_dir():
        raise FileNotFoundError(args.output)
    if args.verify_samples > 0 and not verify(args)["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
