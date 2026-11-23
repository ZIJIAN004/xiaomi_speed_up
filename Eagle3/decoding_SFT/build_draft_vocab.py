#!/usr/bin/env python3
"""Discover the EAGLE-3 draft vocabulary from IDSpace training answers.

Counts exactly the tokens the draft is trained to predict: the ground-truth
assistant answer, serialized and tokenized the same way as
``idspace_gt_dataset.py``. Images are not needed. The size is chosen
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
import random
from collections import Counter
from pathlib import Path

import torch

from idspace_gt_dataset import serialize_target

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
    p.add_argument("--holdout-fraction", type=float, default=0.1,
                   help="rows held out to report coverage on answers not used for selection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_answer_ids(path: Path, tokenizer, max_samples: int):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples:
        rows = rows[:max_samples]
    sequences, failures = [], Counter()
    for row in rows:
        target = row.get("target")
        if not isinstance(target, dict):
            failures["target_not_object"] += 1
            continue
        ids = tokenizer(serialize_target(target), add_special_tokens=False).input_ids
        if not ids:
            failures["empty_answer"] += 1
            continue
        sequences.append(ids)
    if not sequences:
        raise ValueError(f"no usable answers in {path}: {dict(failures)}")
    return len(rows), sequences, failures


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
    if not 0 < args.coverage <= 1 or not 0 <= args.holdout_fraction < 1:
        raise ValueError("--coverage must be in (0, 1]; --holdout-fraction in [0, 1)")
    if args.max_size < 1 or args.round_to < 1:
        raise ValueError("--max-size and --round-to must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    rows, sequences, failures = load_answer_ids(args.train_jsonl, tokenizer, args.max_samples)

    # Selection uses the split-off train part; the held-out part estimates
    # coverage on unseen answers (new names/numbers). Final vocab uses all rows.
    order = list(range(len(sequences)))
    random.Random(args.seed).shuffle(order)
    n_holdout = int(len(order) * args.holdout_fraction)
    holdout = Counter(t for i in order[:n_holdout] for t in sequences[i])
    fit = Counter(t for i in order[n_holdout:] for t in sequences[i])
    full = Counter(t for ids in sequences for t in ids)

    fit_size = choose_size([c for _, c in fit.most_common()], args.coverage, args.round_to, args.max_size)
    fit_vocab = {t for t, _ in fit.most_common(fit_size)}
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
        "rows": rows,
        "usable_answers": len(sequences),
        "failures": dict(failures),
        "answer_tokens": total_tokens,
        "mean_answer_tokens": total_tokens / len(sequences),
        "distinct_tokens": len(full),
        "coverage_target": args.coverage,
        "chosen_size": size,
        "chosen_train_coverage": coverage_of(chosen, full),
        "holdout_rows": n_holdout,
        "holdout_size_selected_without_holdout": fit_size,
        "holdout_coverage": coverage_of(fit_vocab, holdout) if n_holdout else None,
        "train_coverage_by_size": curve,
        "max_draft_id": int(draft_ids.max()),
        "top_tokens": [{"id": t, "count": c, "text": tokenizer.decode([t])} for t, c in full.most_common(50)],
    }
    if report["holdout_coverage"] is not None and report["holdout_coverage"] < args.coverage - 0.005:
        report["warning"] = "held-out coverage is well below target: answers contain many unseen tokens"

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
        "usable_answers", "answer_tokens", "distinct_tokens", "chosen_size",
        "chosen_train_coverage", "holdout_coverage", "train_coverage_by_size")}, ensure_ascii=False, indent=2))
    if "warning" in report:
        print("WARNING:", report["warning"])


if __name__ == "__main__":
    main()
