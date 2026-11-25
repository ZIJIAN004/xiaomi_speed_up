#!/usr/bin/env python3
"""Patch the official DSpark draft vocabulary for the KIE answers (plan A).

The official draft vocabulary (114688 of 248100 target tokens, ordered by
frequency on general text) lacks tokens the fine-tuned target emits in nearly
every answer: compact-JSON separators such as ``":"`` and ``","`` and pieces of
field names (``_birth``, ``_expiry``, ...). A draft cannot propose a token outside
its vocabulary, so every such position ends the accepted run.

The patch keeps the size, hence config.json, every tensor shape and vLLM's
loader, and re-points draft ids at the missing tokens:

* missing = tokens of the training answers (compact JSON + ``<|im_end|>``) not in
  the official vocabulary, most frequent first;
* freed = official draft ids taken from the tail of the official order (its
  rarest tokens), skipping any token that occurs in the training answers and
  special tokens;
* draft id freed[i] now maps to missing[i]. Training then sets its lm_head row
  from the target head and zeroes its Markov row (minicpm_dspark.apply_draft_vocab).

Writes <output>.pt (the patch, for train_dspark --draft-vocab) and <output>.json
(coverage of training and validation answer tokens before/after, and the token
lists). CPU only, tokenizer only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idspace_minicpm_dataset import serialize_target  # noqa: E402
from minicpm_dspark import VOCAB_PATCH_FORMAT, _load_safetensors_dir, vocab_maps  # noqa: E402

END_OF_TURN = "<|im_end|>"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="target model or tokenizer directory")
    p.add_argument("--init-draft", type=Path, required=True, help="official MiniCPM-OCR-1.0-DSpark")
    p.add_argument("--train-jsonl", type=Path, required=True, help="answers that define the needed tokens")
    p.add_argument("--val-jsonl", type=Path, default=None, help="held-out answers, coverage report only")
    p.add_argument("--min-count", type=int, default=1, help="add missing tokens seen at least this often")
    p.add_argument("--output", type=Path, required=True, help="output stem: writes <stem>.pt and <stem>.json")
    return p.parse_args()


def answer_counts(jsonl: Path, tokenizer, eot_id: int) -> Counter:
    counts: Counter = Counter()
    with jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                answer = serialize_target(json.loads(line)["target"])
                counts.update(tokenizer(answer, add_special_tokens=False).input_ids)
                counts[eot_id] += 1
    return counts


def coverage(counts: Counter, vocab: set[int]) -> float:
    total = sum(counts.values())
    return sum(c for t, c in counts.items() if t in vocab) / max(total, 1)


def plan_patch(official_target_ids: torch.Tensor, counts: Counter, protected: set[int],
               min_count: int = 1) -> tuple[torch.Tensor, list[int], list[int]]:
    """(new target ids per draft id, replaced draft ids, added target ids).

    ``protected`` target ids (answer tokens, special tokens) are never freed.
    """
    official = set(official_target_ids.tolist())
    missing = [t for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
               if c >= min_count and t not in official]
    freed: list[int] = []
    for draft_id in range(official_target_ids.numel() - 1, -1, -1):
        if len(freed) == len(missing):
            break
        if int(official_target_ids[draft_id]) not in protected:
            freed.append(draft_id)
    if len(freed) < len(missing):
        raise ValueError(f"only {len(freed)} unprotected draft ids for {len(missing)} missing tokens")
    new_ids = official_target_ids.clone()
    for draft_id, token in zip(freed, missing):
        new_ids[draft_id] = token
    if new_ids.unique().numel() != new_ids.numel():
        raise AssertionError("patched vocabulary maps two draft ids to one target id")
    return new_ids, freed, missing


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eot_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    state = _load_safetensors_dir(args.init_draft)
    base_d2t = state["d2t"]
    verifier_vocab = int(json.loads((args.init_draft / "config.json").read_text(encoding="utf-8"))["vocab_size"])
    official_ids, _ = vocab_maps(base_d2t, verifier_vocab)

    train_counts = answer_counts(args.train_jsonl, tokenizer, eot_id)
    special = set(tokenizer.all_special_ids) | {int(i) for i in tokenizer.get_added_vocab().values()}
    new_ids, freed, added = plan_patch(official_ids, train_counts, set(train_counts) | special, args.min_count)
    d2t = (new_ids - torch.arange(new_ids.numel())).to(base_d2t.dtype)
    vocab_maps(d2t, verifier_vocab)  # same checks the loader runs

    replaced = torch.tensor(freed, dtype=torch.long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": VOCAB_PATCH_FORMAT, "base_d2t": base_d2t.clone(), "d2t": d2t,
                "replaced_draft_ids": replaced, "added_target_ids": torch.tensor(added, dtype=torch.long),
                "removed_target_ids": official_ids[replaced].clone()},
               args.output.with_suffix(".pt"))

    before, after = set(official_ids.tolist()), set(new_ids.tolist())

    def token_rows(ids, counts=None):
        return [{"id": int(t), "token": tokenizer.convert_ids_to_tokens(int(t)),
                 **({"count": counts[int(t)]} if counts is not None else {})} for t in ids]

    report = {
        "init_draft": str(args.init_draft), "model": str(args.model), "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl), "draft_vocab_size": int(new_ids.numel()), "min_count": args.min_count,
        "replaced": len(freed),
        "train_answer_tokens": sum(train_counts.values()), "train_distinct_tokens": len(train_counts),
        "train_coverage_before": coverage(train_counts, before),
        "train_coverage_after": coverage(train_counts, after),
        "added": token_rows(added, train_counts),
        "removed": [{"draft_id": d, **row} for d, row in zip(freed, token_rows(official_ids[replaced].tolist()))],
    }
    if args.val_jsonl:
        val_counts = answer_counts(args.val_jsonl, tokenizer, eot_id)
        report.update({"val_answer_tokens": sum(val_counts.values()),
                       "val_coverage_before": coverage(val_counts, before),
                       "val_coverage_after": coverage(val_counts, after),
                       "val_still_missing": token_rows([t for t in val_counts if t not in after], val_counts)})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    summary = {k: v for k, v in report.items() if not isinstance(v, list)}
    print(json.dumps(summary, indent=2), flush=True)
    print("added (top 20): " + " ".join(f"{r['token']!r}x{r['count']}" for r in report["added"][:20]), flush=True)
    print(f"wrote {args.output.with_suffix('.pt')} and .json", flush=True)


if __name__ == "__main__":
    main()
