#!/usr/bin/env python3
"""Grouping A: always-on shared set + capped spherical k-means of the draft lm_head rows.

1. Shared set = the ``--shared`` most frequent draft ids in the TRAINING answers
   (always in every shortlist; DeepSeekMoE-style shared expert). ``--shared 0`` = none.
2. The other draft ids are grouped by their lm_head rows (balanced_kmeans.py):
   ``--num-groups`` groups of at most ceil(N/K * (1 + slack)) tokens each.

Writes <output>.pt (vocab_common grouping file, read by block_coverage.py and,
later, the router) and <output>.json (settings, group sizes, k-means history and
the decoded tokens of the most used groups for a manual look). CPU by default.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balanced_kmeans import balanced_spherical_kmeans, capacity  # noqa: E402
from vocab_common import (  # noqa: E402
    SHARED, answer_sequences, draft_counts, inverse_lookup, load_draft_head, load_draft_vocab, save_groups,
    to_draft,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--official-draft", type=Path, required=True, help="official MiniCPM-OCR-1.0-DSpark dir")
    p.add_argument("--draft-vocab", type=Path, default=None, help="KIE vocab patch (build_kie_draft_vocab.py)")
    p.add_argument("--target-model", type=Path, required=True,
                   help="target model dir: tokenizer and the target head rows of patched draft ids")
    p.add_argument("--head-source", choices=("checkpoint", "verifier"), default="checkpoint",
                   help="draft head rows to cluster: as the warm-started draft serves them, or the target head")
    p.add_argument("--train-jsonl", type=Path, required=True, help="answers that rank the shared set")
    p.add_argument("--num-groups", type=int, required=True)
    p.add_argument("--shared", type=int, default=2048, help="size of the always-on shared set (0 = none)")
    p.add_argument("--slack", type=float, default=0.1, help="group cap = ceil(N/K * (1 + slack))")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--topk", type=int, default=16, help="candidate centroids per row in the capped assignment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-rows", type=int, default=0, help="read only the first N training answers (smoke)")
    p.add_argument("--output", type=Path, required=True, help="output stem: writes <stem>.pt and <stem>.json")
    args = p.parse_args(argv)
    if args.num_groups < 1 or args.shared < 0 or args.iters < 1 or args.topk < 1:
        p.error("num-groups, iters, topk must be >= 1 and shared >= 0")
    return args


def choose_shared(counts: np.ndarray, size: int) -> np.ndarray:
    """Most frequent draft ids (ties by id), only ids that occur at all."""
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(counts.size), -counts))
    order = order[counts[order] > 0]
    return np.sort(order[:size])


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    vocab = load_draft_vocab(args.official_draft, args.draft_vocab)
    target_ids = vocab["target_ids"]
    draft_vocab = target_ids.numel()
    head = load_draft_head(args.official_draft, vocab, args.target_model, args.head_source)

    train = to_draft(answer_sequences(args.train_jsonl, tokenizer, args.max_rows),
                     inverse_lookup(target_ids, vocab["verifier_vocab"]))
    counts = draft_counts(train, draft_vocab)
    shared = choose_shared(counts, args.shared)
    rest = np.setdiff1d(np.arange(draft_vocab), shared)
    print(f"draft vocab {draft_vocab}, shared {shared.size}, clustered {rest.size} into {args.num_groups} groups "
          f"(cap {capacity(rest.size, args.num_groups, args.slack)})", flush=True)

    labels, history = balanced_spherical_kmeans(
        head[rest], args.num_groups, iters=args.iters, slack=args.slack, seed=args.seed, topk=args.topk,
        device=args.device)
    token_group = np.full(draft_vocab, SHARED, dtype=np.int64)
    token_group[rest] = labels
    sizes = np.bincount(labels, minlength=args.num_groups)

    total = int(counts.sum())
    group_mass = np.bincount(labels, weights=counts[rest], minlength=args.num_groups)
    top_groups = np.argsort(-group_mass)[:20]

    def decode(ids):
        return [tokenizer.convert_ids_to_tokens(int(target_ids[i])) for i in ids]

    meta = {
        "method": "A_lm_head_kmeans", "official_draft": str(args.official_draft),
        "draft_vocab": str(args.draft_vocab) if args.draft_vocab else None,
        "target_model": str(args.target_model), "head_source": args.head_source,
        "train_jsonl": str(args.train_jsonl), "train_answers": len(train), "num_groups": args.num_groups,
        "shared": int(shared.size), "slack": args.slack, "iters": args.iters, "topk": args.topk,
        "seed": args.seed,
    }
    save_groups(args.output.with_suffix(".pt"), token_group, args.num_groups, target_ids,
                "A_lm_head_kmeans", meta)
    report = {
        **meta,
        "train_answer_tokens": total,
        "train_token_mass_shared": float(counts[shared].sum() / max(total, 1)),
        "group_size": {"min": int(sizes.min()), "mean": float(sizes.mean()), "max": int(sizes.max())},
        "kmeans_history": history,
        "seconds": time.perf_counter() - started,
        "shared_tokens_head": decode(shared[np.argsort(-counts[shared])][:50]),
        "top_groups_by_train_mass": [
            {"group": int(g), "size": int(sizes[g]), "train_mass": float(group_mass[g] / max(total, 1)),
             "tokens_by_count": decode(rest[labels == g][np.argsort(-counts[rest[labels == g]])][:24])}
            for g in top_groups],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(f"shared set holds {report['train_token_mass_shared']:.1%} of training answer tokens; "
          f"group sizes {report['group_size']}", flush=True)
    print(f"wrote {args.output.with_suffix('.pt')} and .json ({report['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
