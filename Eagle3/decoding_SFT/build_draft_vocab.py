#!/usr/bin/env python3
"""Discover the EAGLE-3 draft vocabulary from IDSpace training answers.

Counts exactly the tokens the draft is trained to predict: the ground-truth
assistant answer plus its ``<|im_end|>``, serialized and tokenized the same
way as ``idspace_gt_dataset.py``. Images are not needed. The size is chosen
automatically as the smallest frequency-ranked prefix that reaches
``--coverage``, rounded up to ``--round-to`` and capped by ``--max-size``.

Output ``draft_vocab.pt`` holds ``draft_ids`` (target token ids, ascending)
plus the official EAGLE-3 ``d2t`` offsets; ``train_eagle_idspace.py
--draft-vocab`` builds ``t2d`` from it with the loaded target vocab size.

Remote example:
    python Eagle3/decoding_SFT/build_draft_vocab.py \
      --train-jsonl /home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl \
      --tokenizer /home/mr_jiang/models/xiaomi-ocr-0/model \
      --output-dir /home/mr_jiang/qkr/runs/idspace_e3/draft_vocab
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from idspace_gt_dataset import end_of_turn_id, serialize_target, split_train_val

REPORT_SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384, 32000)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="model or processor directory")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--coverage", type=float, default=0.999,
                   help="target frequency-weighted coverage of answer tokens")
    p.add_argument("--max-size", type=int, default=32000, help="official EAGLE-3 draft vocab size")
    p.add_argument("--round-to", type=int, default=128)
    # Must match train_eagle_idspace.py so validation rows never shape the vocab.
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_answer_ids(path: Path, tokenizer, max_samples: int):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    # One entry per row (None for unusable rows) so split indices stay aligned.
    sequences, failures = [], Counter()
    eot = end_of_turn_id(tokenizer)  # supervised after every answer, see idspace_gt_dataset
    for row in rows:
        target = row.get("target")
        ids = tokenizer(serialize_target(target), add_special_tokens=False).input_ids             if isinstance(target, dict) else None
        if ids is None:
            failures["target_not_object"] += 1
        elif not ids:
            failures["empty_answer"] += 1
        sequences.append(ids + [eot] if ids else None)
    if not any(sequences):
        raise ValueError(f"no usable answers in {path}: {dict(failures)}")
    return sequences, failures


def choose_size(ranked_counts: list[int], coverage: float, round_to: int, max_size: int) -> int:
    total = sum(ranked_counts)
    running, needed = 0, len(ranked_counts)
    for rank, count in enumerate(ranked_counts, start=1):
        running += count
        if running / total >= coverage:
            needed = rank
            break
    rounded = math.ceil(needed / round_to) * round_to
    return min(rounded, len(ranked_counts), max_size)


def coverage_of(ids: set[int], counter: Counter) -> float:
    total = sum(counter.values())
    return sum(c for token, c in counter.items() if token in ids) / total if total else float("nan")


def main():
    args = parse_args()
    if not 0 < args.coverage <= 1:
        raise ValueError("--coverage must be in (0, 1]")
    if args.max_size < 1 or args.round_to < 1:
        raise ValueError("--max-size and --round-to must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    sequences, failures = load_answer_ids(args.train_jsonl, tokenizer, args.max_samples)

    # Vocab is chosen on the TRAINING split only; the validation split (same
    # seeded split as training) measures coverage on unseen answers.
    train_idx, val_idx = split_train_val(len(sequences), args.val_fraction, args.seed)
    full = Counter(t for i in train_idx if sequences[i] for t in sequences[i])
    val = Counter(t for i in val_idx if sequences[i] for t in sequences[i])
    usable = sum(1 for i in train_idx if sequences[i])
    if not full:
        raise ValueError("training split has no usable answers")

    size = choose_size([c for _, c in full.most_common()], args.coverage, args.round_to, args.max_size)
    draft_ids = torch.tensor(sorted(t for t, _ in full.most_common(size)), dtype=torch.long)
    chosen = set(draft_ids.tolist())

    total_tokens = sum(full.values())
    curve = {}
    ranked = [c for _, c in full.most_common()]
    for n in REPORT_SIZES:
        curve[str(n)] = sum(ranked[:n]) / total_tokens
    report = {
        "train_jsonl": str(args.train_jsonl),
        "tokenizer": str(args.tokenizer),
        "rows": len(sequences),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "usable_train_answers": usable,
        "failures": dict(failures),
        "train_answer_tokens": total_tokens,
        "mean_answer_tokens": total_tokens / usable,
        "distinct_tokens": len(full),
        "coverage_target": args.coverage,
        "chosen_size": size,
        "chosen_train_coverage": coverage_of(chosen, full),
        "val_coverage": coverage_of(chosen, val) if val else None,
        "train_coverage_by_size": curve,
        "max_draft_id": int(draft_ids.max()),
        "top_tokens": [{"id": t, "count": c, "text": tokenizer.decode([t])} for t, c in full.most_common(50)],
    }
    if report["val_coverage"] is not None and report["val_coverage"] < args.coverage - 0.005:
        report["warning"] = "validation coverage is well below target: answers contain many unseen tokens"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "draft_ids": draft_ids,
        "d2t": draft_ids - torch.arange(len(draft_ids)),  # official EAGLE-3 offset format
        "tokenizer_len": len(tokenizer),
        "report": report,
    }, args.output_dir / "draft_vocab.pt")
    (args.output_dir / "draft_vocab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "rows", "train_rows", "val_rows", "usable_train_answers", "mean_answer_tokens",
        "distinct_tokens", "chosen_size", "chosen_train_coverage", "val_coverage",
        "train_coverage_by_size")}, ensure_ascii=False, indent=2))
    if "warning" in report:
        print("WARNING:", report["warning"])


if __name__ == "__main__":
    main()
