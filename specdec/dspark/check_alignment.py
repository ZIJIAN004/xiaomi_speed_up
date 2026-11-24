#!/usr/bin/env python3
"""Alignment gate: does our training pipeline reproduce what the official draft sees?

Run on the ORIGINAL MiniCPM-OCR 1.0 with the official DSpark draft, before any
training. Checks:

1. Round trip: official weights -> speculators model -> save_official gives
   bit-identical tensors and the same config.json (so trained drafts load in vLLM).
2. Features: aux hidden states / pre-norm last hidden state are the same through
   the base model and the full model, and the reconstructed verifier targets
   (verifier_norm + reduced verifier head) match the target's real logits.
3. Positions: the target's text model gets plain 0..T-1 positions (as in vLLM).
4. Metrics: DSpark eal / accept_len / per-position accuracy of the official draft
   on held-out IDSpace samples. Compare eal with the mean acceptance length that
   vLLM reports for the same draft and data (see README).

Writes <output-dir>/alignment_report.json and exits non-zero if a hard check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idspace_minicpm_dataset import collate_single  # noqa: E402
from minicpm_dspark import _load_safetensors_dir, save_official  # noqa: E402
from train_dspark import Pipeline, build_datasets, evaluate, parse_args as train_parse_args, print_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="original MiniCPM-OCR 1.0")
    p.add_argument("--init-draft", type=Path, required=True, help="official MiniCPM-OCR-1.0-DSpark")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--text-first", action="store_true", help="default: image first (official order)")
    p.add_argument("--feature-samples", type=int, default=4, help="samples for the feature/target checks")
    p.add_argument("--eval-samples", type=int, default=64, help="held-out samples for eal/accept_len")
    p.add_argument("--max-anchors", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-jsonl", type=Path, default=None,
                   help="IDSpace validation.jsonl (same rows as the vLLM benchmark); else a seeded "
                        "--val-fraction of --train-jsonl")
    p.add_argument("--val-fraction", type=float, default=0.05)
    return p.parse_args()


def round_trip(pipe, report):
    original = _load_safetensors_dir(pipe.draft.source_dir)
    with tempfile.TemporaryDirectory() as tmp:
        out = save_official(pipe.draft, Path(tmp) / "draft")
        saved = _load_safetensors_dir(out)
        same_config = (out / "config.json").read_bytes() == (pipe.draft.source_dir / "config.json").read_bytes()
    mismatched = sorted(k for k in original if k not in saved or saved[k].dtype != original[k].dtype
                        or not torch.equal(saved[k], original[k]))
    report["round_trip"] = {"keys": len(original), "extra_keys": sorted(set(saved) - set(original)),
                            "mismatched": mismatched, "config_identical": same_config}
    return not mismatched and not report["round_trip"]["extra_keys"] and same_config


@torch.no_grad()
def lm_head_vs_verifier(pipe, report):
    model = pipe.model
    diff = (model.lm_head.weight.float() - model.verifier_lm_head.weight.float()).abs()
    report["lm_head_vs_target_head"] = {
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
        "note": "≈0 means the official draft head is the target head (frozen); then retrain with "
                "--lm-head-init verifier so it follows the fine-tuned target",
    }


@torch.no_grad()
def feature_checks(pipe, loader, limit, report):
    worst = {"aux_base_vs_full_rel": 0.0, "last_base_vs_full": 0.0, "target_logit_max_abs": 0.0}
    agree = total = 0
    target_ids = pipe.draft.target_ids.to(pipe.device)
    model = pipe.model
    for n, raw in enumerate(loader):
        if n >= limit:
            break
        inputs = {k: v.to(pipe.device) for k, v in raw["inputs"].items()}
        base = pipe.extractor(inputs)                       # also checks sequential positions
        full = pipe.extractor(inputs, runner=pipe.target)
        # Relative to the feature scale: base and full model may pick different kernels.
        worst["aux_base_vs_full_rel"] = max(worst["aux_base_vs_full_rel"], float(
            (base["hidden_states"].float() - full["hidden_states"].float()).abs().max()
            / full["hidden_states"].float().abs().max().clamp_min(1e-6)))
        worst["last_base_vs_full"] = max(worst["last_base_vs_full"], float(
            (base["verifier_last_hidden_states"].float() - full["verifier_last_hidden_states"].float()).abs().max()))
        pos = raw["loss_mask"][0].nonzero().squeeze(-1).to(pipe.device)
        real = full["logits"][0, pos].float().index_select(-1, target_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rebuilt = model.verifier_lm_head(model.verifier_norm(base["verifier_last_hidden_states"][0, pos])).float()
        worst["target_logit_max_abs"] = max(worst["target_logit_max_abs"], float((real - rebuilt).abs().max()))
        agree += int((real.argmax(-1) == rebuilt.argmax(-1)).sum())
        total += pos.numel()
    worst["target_argmax_agreement"] = agree / max(total, 1)
    worst["checked_positions"] = total
    report["features"] = worst
    return worst["aux_base_vs_full_rel"] < 1e-2 and worst["target_argmax_agreement"] >= 0.999


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    targs = train_parse_args([
        "--model", str(args.model), "--init-draft", str(args.init_draft),
        "--train-jsonl", str(args.train_jsonl), "--image-root", str(args.image_root),
        "--output-dir", str(args.output_dir), "--max-anchors", str(args.max_anchors),
        "--seed", str(args.seed), "--val-fraction", str(args.val_fraction), "--lm-head-init", "checkpoint",
        *(["--val-jsonl", str(args.val_jsonl)] if args.val_jsonl else []),
        *(["--prompt-file", str(args.prompt_file)] if args.prompt_file else []),
        *(["--text-first"] if args.text_first else []),
    ])
    pipe = Pipeline(targs, torch.device("cuda:0"))
    _, val_set = build_datasets(targs, pipe.processor)
    if val_set is None:
        raise ValueError("need --val-jsonl or --val-fraction > 0")
    loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_single)

    report = {"model": str(args.model), "init_draft": str(args.init_draft)}
    checks = {"round_trip": round_trip(pipe, report)}
    lm_head_vs_verifier(pipe, report)
    checks["features"] = feature_checks(pipe, loader, args.feature_samples, report)
    metrics = evaluate(pipe, loader, args.seed, args.eval_samples)
    print_metrics("official draft", metrics)
    report["metrics"] = metrics
    report["checks"] = checks
    path = args.output_dir / "alignment_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2), flush=True)
    print(f"wrote {path}", flush=True)
    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
