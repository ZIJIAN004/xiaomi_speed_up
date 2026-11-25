#!/usr/bin/env python3
"""Token-length statistics of the DSpark training data (processor only, no model, CPU).

Renders every IDSpace row exactly as train_dspark.py does and reports, per split,
the distribution of total sequence length, image tokens, answer tokens and DSpark
anchor positions, the share of samples whose anchors exceed --max-anchors, and the
disk size an offline feature cache would need. Answers two questions:
  * is --max-anchors large enough (anchors_over_max should be ~0)?
  * would caching target features to disk be affordable?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idspace_minicpm_dataset import add_data_args, collate_single, make_datasets_factory  # noqa: E402

IMAGE_TOKEN = "<|image_pad|>"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="model or processor directory")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, default=None)
    p.add_argument("--image-root", type=Path, required=True)
    add_data_args(p)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--block-size", type=int, default=15)
    p.add_argument("--max-anchors", type=int, default=256)
    p.add_argument("--aux-layers", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    return p.parse_args()


def percentiles(values: list[int]) -> dict[str, float]:
    t = torch.tensor(values, dtype=torch.float64)
    qs = torch.quantile(t, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)).tolist()
    return {"mean": round(t.mean().item(), 1), "p50": qs[0], "p90": qs[1], "p99": qs[2],
            "min": int(t.min()), "max": int(t.max())}


def measure(dataset, args, image_id, hidden_size) -> dict:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single,
                        num_workers=args.workers)
    rows = {"seq": [], "image": [], "answer": [], "anchors": [], "prefix": []}
    first_ids = None
    shared_prefix = None
    for sample in loader:
        ids = sample["inputs"]["input_ids"][0]
        seq_len = ids.numel()
        rows["seq"].append(seq_len)
        rows["answer"].append(int(sample["assistant_mask"].sum()))
        rows["anchors"].append(int(sample["loss_mask"][0, :-args.block_size].sum()))
        first = seq_len
        if image_id is not None:
            is_image = ids == image_id
            rows["image"].append(int(is_image.sum()))
            first = int(is_image.nonzero()[0]) if bool(is_image.any()) else seq_len
            rows["prefix"].append(first)
        # Longest token prefix shared by every sample (identical prefix => identical features).
        # It stops at the first image token: image placeholder ids are identical across
        # samples, but their embeddings come from different pictures.
        ids_list = ids.tolist()
        if first_ids is None:
            first_ids, shared_prefix = ids_list, first
        else:
            limit = min(shared_prefix, len(ids_list), first)
            match = 0
            while match < limit and ids_list[match] == first_ids[match]:
                match += 1
            shared_prefix = match
    n = len(rows["seq"])
    if n == 0:
        return {"samples": 0}
    per_token_aux = args.aux_layers * hidden_size * 2          # bf16 aux features
    per_token_last = hidden_size * 2                           # bf16 pre-norm last hidden
    cache_bytes = [(s - shared_prefix) * per_token_aux + (a + args.block_size) * per_token_last
                   for s, a in zip(rows["seq"], rows["answer"])]
    report = {
        "samples": n,
        "seq_len": percentiles(rows["seq"]),
        "answer_tokens_incl_eot": percentiles(rows["answer"]),
        "anchor_positions": percentiles(rows["anchors"]),
        "anchors_over_max_frac": sum(a > args.max_anchors for a in rows["anchors"]) / n,
        "shared_prefix_tokens": shared_prefix,
        "offline_cache_mb_per_sample": percentiles([round(b / 2**20) for b in cache_bytes]),
        "offline_cache_total_gb": round(sum(cache_bytes) / 2**30, 2),
    }
    if rows["image"]:
        report["image_tokens"] = percentiles(rows["image"])
        report["tokens_before_first_image"] = percentiles(rows["prefix"])
    return report


def main():
    args = parse_args()
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    hidden_size = int(getattr(config, "text_config", config).hidden_size)
    image_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    if image_id is None or image_id == processor.tokenizer.unk_token_id:
        print(f"note: tokenizer has no {IMAGE_TOKEN}; image token counts skipped", flush=True)
        image_id = None
    make = make_datasets_factory(args, processor)
    report = {"model": str(args.model), "task": args.task, "layout_jsonl": str(args.layout_jsonl),
              "image_first": not args.text_first, "max_pixels": args.max_pixels,
              "max_slice_nums": args.max_slice_nums, "max_anchors": args.max_anchors,
              "block_size": args.block_size, "hidden_size": hidden_size}
    splits = [("train", args.train_jsonl)] + ([("val", args.val_jsonl)] if args.val_jsonl else [])
    for name, path in splits:
        dataset = make(path, args.max_samples)
        report[name] = measure(dataset, args, image_id, hidden_size)
        print(name, json.dumps(report[name], indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
