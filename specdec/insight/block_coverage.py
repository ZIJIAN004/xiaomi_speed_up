#!/usr/bin/env python3
"""Oracle shortlist size of block-shared vocabulary routing (measurement 1).

A DSpark round drafts a block of 15 tokens. With block routing the draft scores
only a shortlist = always-on shared set + the groups a router picks for the WHOLE
block. Here the router is an oracle: it picks exactly the groups holding the
block's target tokens. So the numbers are the smallest shortlist each grouping
allows, i.e. the upper bound on what routing can save, before training a router.

Blocks: every answer position p starts one block = answer[p : p + block_size]
(slot k predicts token p+k+1 from anchor p; the first block's anchor is the last
prompt token), truncated at the end of the answer, as training anchors are.

Per grouping file and per prefix length m (only the first m slots must be
covered; later slots rarely get accepted):

* shortlist = |shared| + sizes of the distinct groups among the block's first m
  in-vocabulary tokens (draft lm_head rows read for that block);
* groups = how many groups that takes;
* shared_only = blocks that need no group at all (a static shortlist = the shared
  set would cover them: the FR-Spec / VocabTrim baseline);
* oov = blocks with a token outside the draft vocabulary (no shortlist helps).

Per-position oracle (DynaSpec-style, one choice per slot): mean over tokens of
|shared| + size of the token's own group. Block sharing pays the difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab_common import SHARED, answer_sequences, inverse_lookup, load_groups, to_draft  # noqa: E402

PAD = -3       # past the end of the answer
OOV_GROUP = -2  # target token outside the draft vocabulary


def block_windows(sequences: list[np.ndarray], block: int) -> np.ndarray:
    """[sum of lengths, block] windows starting at every position, PAD past the end."""
    rows = []
    for seq in sequences:
        padded = np.concatenate([seq, np.full(block, PAD, dtype=np.int64)])
        rows.append(np.lib.stride_tricks.sliding_window_view(padded, block)[:seq.size])
    return np.concatenate(rows).copy()


def window_groups(windows: np.ndarray, token_group: np.ndarray) -> np.ndarray:
    """Draft-id windows -> group ids (>= 0), SHARED, OOV_GROUP or PAD."""
    groups = np.full(windows.shape, PAD, dtype=np.int64)
    real = windows != PAD
    oov = real & (windows < 0)
    inside = real & ~oov
    groups[oov] = OOV_GROUP
    groups[inside] = token_group[windows[inside]]
    return groups


def block_shortlists(groups: np.ndarray, sizes: np.ndarray, shared_size: int, m: int) -> dict[str, np.ndarray]:
    """Per block: shortlist size, groups used, OOV flag (first m slots)."""
    part = np.sort(groups[:, :m], axis=1)
    distinct = part >= 0
    distinct[:, 1:] &= part[:, 1:] != part[:, :-1]
    group_rows = np.where(distinct, sizes[np.clip(part, 0, None)], 0).sum(axis=1)
    return {"shortlist": shared_size + group_rows, "groups": distinct.sum(axis=1),
            "oov": (groups[:, :m] == OOV_GROUP).any(axis=1)}


def distribution(values: np.ndarray, vocab: int) -> dict[str, float]:
    values = values.astype(np.float64)
    out = {"mean": float(values.mean())}
    for q in (50, 90, 99):
        out[f"p{q}"] = float(np.percentile(values, q))
    out["max"] = float(values.max())
    out.update({f"{k}_frac": v / vocab for k, v in list(out.items())})
    return out


def coverage_report(draft_sequences: list[np.ndarray], token_group: np.ndarray, num_groups: int,
                    block: int, prefixes: list[int]) -> dict:
    vocab = token_group.size
    sizes = np.bincount(token_group[token_group >= 0], minlength=num_groups)
    shared_size = int((token_group == SHARED).sum())
    groups = window_groups(block_windows(draft_sequences, block), token_group)

    tokens = np.concatenate(draft_sequences)
    in_vocab = tokens >= 0
    token_groups = token_group[tokens[in_vocab]]
    per_position = shared_size + np.where(token_groups >= 0, sizes[np.clip(token_groups, 0, None)], 0)

    report = {"draft_vocab": int(vocab), "num_groups": int(num_groups), "shared": shared_size,
              "group_size_max": int(sizes.max()), "blocks": int(groups.shape[0]), "tokens": int(tokens.size),
              "token_oov_frac": float((~in_vocab).mean()),
              "token_shared_frac": float((token_groups == SHARED).mean()) if token_groups.size else 0.0,
              "per_position_oracle": distribution(per_position, vocab), "prefix": {}}
    for m in prefixes:
        stats = block_shortlists(groups, sizes, shared_size, min(m, block))
        clean = ~stats["oov"]
        if not clean.any():
            raise ValueError(f"every block has an out-of-vocabulary token within {m} slots")
        report["prefix"][str(m)] = {
            "shortlist": distribution(stats["shortlist"][clean], vocab),
            "groups": distribution(stats["groups"][clean], vocab),
            "shared_only_frac": float((stats["groups"][clean] == 0).mean()),
            "oov_block_frac": float(stats["oov"].mean()),
        }
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", type=Path, nargs="+", required=True, help="grouping .pt files")
    p.add_argument("--val-jsonl", type=Path, required=True, help="held-out answers")
    p.add_argument("--tokenizer", type=Path, required=True, help="target model or tokenizer dir")
    p.add_argument("--block-size", type=int, default=15)
    p.add_argument("--prefixes", default="4,8,12,15", help="comma-separated slot counts that must be covered")
    p.add_argument("--max-rows", type=int, default=0, help="read only the first N answers (smoke)")
    p.add_argument("--output", type=Path, required=True, help="coverage .json")
    args = p.parse_args(argv)
    args.prefixes = sorted({int(x) for x in args.prefixes.split(",") if x.strip()})
    if args.block_size < 1 or not args.prefixes or min(args.prefixes) < 1:
        p.error("block-size and every prefix must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    sequences = answer_sequences(args.val_jsonl, tokenizer, args.max_rows)
    results = {"val_jsonl": str(args.val_jsonl), "answers": len(sequences), "block_size": args.block_size,
               "groupings": {}}
    for path in args.groups:
        grouping = load_groups(path)
        target_ids = grouping["target_ids"]
        verifier_vocab = max(len(tokenizer), int(target_ids.max()) + 1)
        draft = to_draft(sequences, inverse_lookup(target_ids, verifier_vocab))
        report = coverage_report(draft, grouping["token_group"], grouping["num_groups"], args.block_size,
                                 args.prefixes)
        report.update(method=grouping["method"], meta=grouping["meta"])
        results["groupings"][path.stem] = report
        pos = report["per_position_oracle"]
        print(f"{path.stem}: K={report['num_groups']} shared={report['shared']} "
              f"token_oov={report['token_oov_frac']:.4%} per-position mean={pos['mean_frac']:.2%}", flush=True)
        for m, row in report["prefix"].items():
            s = row["shortlist"]
            print(f"  prefix {m:>2}: shortlist p50={s['p50_frac']:.2%} p90={s['p90_frac']:.2%} "
                  f"p99={s['p99_frac']:.2%} groups p90={row['groups']['p90']:.0f} "
                  f"shared_only={row['shared_only_frac']:.1%} oov_blocks={row['oov_block_frac']:.2%}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
