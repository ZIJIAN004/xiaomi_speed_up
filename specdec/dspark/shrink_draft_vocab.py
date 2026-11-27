#!/usr/bin/env python3
"""Shrink a DSpark draft to a task vocabulary (static trimming, VocabTrim-style).

The whole_card answers use 1899 distinct tokens (1.66 % of the 114688-token draft
vocabulary); those cover 99.996 % of validation answer tokens and 99.94 % of all
15-token blocks. A draft cannot propose a token outside its vocabulary, so every
other row of ``lm_head`` and of the Markov ``W2`` is computed on every slot and
never wins. This writes a complete official-layout draft directory whose
vocabulary is:

* every target token of the TRAINING answers seen at least ``--min-count`` times
  (compact JSON + ``<|im_end|>``, as build_kie_draft_vocab.py counts them);
* the tokenizer's special tokens that the source draft vocabulary holds;
* padding up to a multiple of ``--align`` (GEMM-friendly) with the source draft's
  own most frequent remaining ids (the official order is by frequency).

Rows: a token the source draft already scores keeps its ``lm_head`` and
``markov_w2`` rows bit-exact. A token the source lacks, or one re-pointed by a
``--draft-vocab`` patch, takes the target head row and a zero Markov row, exactly
as minicpm_dspark.apply_draft_vocab + attach_verifier do. ``markov_w1`` (indexed
by target id), the backbone and the confidence head are copied unchanged.
Draft ids are ordered by target id. Only ``config.json:draft_vocab_size`` changes;
vLLM sizes ``lm_head`` and ``markov_w2`` from it.

Use the output as ``--init-draft`` WITHOUT ``--draft-vocab`` (the patch is folded in).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kie_draft_vocab import END_OF_TURN, answer_counts  # noqa: E402
from idspace_minicpm_dataset import serialize_target  # noqa: E402
from minicpm_dspark import VOCAB_PATCH_FORMAT, _load_safetensors_dir, vocab_maps  # noqa: E402

VOCAB_KEYS = ("d2t", "lm_head.weight", "markov_head.markov_w2.weight")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft", type=Path, required=True, help="source official-layout draft dir")
    p.add_argument("--draft-vocab", type=Path, default=None, help="vocab patch the source is used with, if any")
    p.add_argument("--target-model", type=Path, required=True,
                   help="tokenizer + target head rows for tokens the source draft does not score")
    p.add_argument("--train-jsonl", type=Path, required=True, help="answers that define the vocabulary")
    p.add_argument("--val-jsonl", type=Path, default=None, help="held-out answers, coverage report only")
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--align", type=int, default=64, help="pad the vocabulary size to a multiple of this")
    p.add_argument("--block-size", type=int, default=15, help="block coverage in the report")
    p.add_argument("--output", type=Path, required=True, help="new draft dir (must not exist)")
    args = p.parse_args(argv)
    if args.min_count < 1 or args.align < 1:
        p.error("min-count and align must be >= 1")
    return args


def source_vocab(state: dict, verifier_vocab: int, patch_path: Path | None) -> tuple[torch.Tensor, set[int]]:
    """(target id per source draft id, draft ids whose rows belong to another token)."""
    d2t = state["d2t"]
    stale: set[int] = set()
    if patch_path is not None:
        patch = torch.load(patch_path, map_location="cpu", weights_only=True)
        if patch.get("format") != VOCAB_PATCH_FORMAT:
            raise ValueError(f"{patch_path}: not a {VOCAB_PATCH_FORMAT} file")
        if not torch.equal(patch["base_d2t"].long(), d2t.long()):
            raise ValueError(f"{patch_path} was built from another draft vocabulary than the source")
        d2t = patch["d2t"]
        stale = set(patch["replaced_draft_ids"].long().tolist())
    target_ids, _ = vocab_maps(d2t, verifier_vocab)
    return target_ids, stale


def read_target_head(model_dir: Path) -> torch.Tensor:
    """The target's output head (tied to embed_tokens in MiniCPM-OCR), read without loading the model."""
    from safetensors import safe_open

    for suffix in ("lm_head.weight", "language_model.embed_tokens.weight", "embed_tokens.weight"):
        for path in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(path), "pt") as stream:
                for key in stream.keys():
                    if key.endswith(suffix):
                        return stream.get_tensor(key)
    raise KeyError(f"no output head / embed_tokens tensor in {model_dir}")


def plan_vocab(counts: Counter, source_ids: torch.Tensor, special: set[int], min_count: int,
               align: int) -> tuple[list[int], dict[str, list[int]]]:
    """Sorted target ids of the new vocabulary, and where each came from."""
    needed = {t for t, c in counts.items() if c >= min_count}
    in_source = set(source_ids.tolist())
    keep = needed | (special & in_source)
    size = -(-len(keep) // align) * align
    padding = []
    for t in source_ids.tolist():      # source order = official frequency order
        if len(keep) + len(padding) >= size:
            break
        if t not in keep:
            padding.append(t)
    vocab = sorted(keep | set(padding))
    return vocab, {"answer": sorted(needed), "special": sorted((special & in_source) - needed),
                   "padding": padding, "not_in_source": sorted(needed - in_source)}


def shrink_state(state: dict, source_ids: torch.Tensor, stale: set[int], vocab: list[int],
                 target_head: torch.Tensor | None) -> tuple[dict, list[int]]:
    """New tensors: source rows where valid, else target head row + zero Markov row."""
    n_source = source_ids.numel()
    other = sorted(k for k, v in state.items() if k not in VOCAB_KEYS and v.ndim and v.shape[0] == n_source)
    if other:
        raise ValueError(f"tensors sized like the draft vocab that this tool does not handle: {other}")
    row_of = {int(t): i for i, t in enumerate(source_ids.tolist()) if i not in stale}
    head, w2 = state["lm_head.weight"], state["markov_head.markov_w2.weight"]
    new_head = torch.empty((len(vocab), head.shape[1]), dtype=head.dtype)
    new_w2 = torch.zeros((len(vocab), w2.shape[1]), dtype=w2.dtype)
    fresh = []
    for i, t in enumerate(vocab):
        row = row_of.get(t)
        if row is None:
            if target_head is None:
                raise ValueError("tokens outside the source rows need the target head")
            new_head[i] = target_head[t].to(head.dtype)
            fresh.append(t)
        else:
            new_head[i] = head[row]
            new_w2[i] = w2[row]
    ids = torch.tensor(vocab, dtype=torch.long)
    out = dict(state)
    out["d2t"] = (ids - torch.arange(len(vocab))).to(state["d2t"].dtype)
    out["lm_head.weight"] = new_head
    out["markov_head.markov_w2.weight"] = new_w2
    out.pop("t2d", None)               # derived from d2t on load
    return out, fresh


def coverage(jsonl: Path, tokenizer, vocab: set[int], block: int) -> dict[str, float]:
    eot = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    tokens = covered = blocks = blocks_ok = 0
    with jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            ids = tokenizer(serialize_target(json.loads(line)["target"]), add_special_tokens=False).input_ids + [eot]
            inside = np.fromiter((t in vocab for t in ids), dtype=bool, count=len(ids))
            tokens += inside.size
            covered += int(inside.sum())
            # block at p covers slots p..p+block-1: all inside <=> next miss is >= block away
            blocks += inside.size
            miss = np.flatnonzero(~inside)
            if miss.size == 0:
                blocks_ok += inside.size
                continue
            starts = np.arange(inside.size)
            nxt = np.searchsorted(miss, starts)
            dist = np.where(nxt < miss.size, miss[np.minimum(nxt, miss.size - 1)] - starts, inside.size)
            blocks_ok += int((dist >= np.minimum(block, inside.size - starts)).sum())
    return {"tokens": tokens, "token_coverage": covered / max(tokens, 1), "blocks": blocks,
            "block_coverage": blocks_ok / max(blocks, 1)}


def write_draft_dir(source: Path, output: Path, config: dict, state: dict, report: dict) -> None:
    """Official layout: non-weight files of the source, new config, one safetensors file."""
    from safetensors.torch import save_file

    if output.exists():
        raise FileExistsError(output)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
            shutil.copy2(item, tmp / item.name)
    config = dict(config, draft_vocab_size=int(state["d2t"].numel()))
    (tmp / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    (tmp / "shrink_info.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.rename(output)


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.draft / "config.json").read_text(encoding="utf-8"))
    verifier_vocab = int(config["vocab_size"])
    state = _load_safetensors_dir(args.draft)
    source_ids, stale = source_vocab(state, verifier_vocab, args.draft_vocab)
    if int(config["draft_vocab_size"]) != source_ids.numel():
        raise ValueError("config draft_vocab_size disagrees with d2t")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    counts = answer_counts(args.train_jsonl, tokenizer, tokenizer.convert_tokens_to_ids(END_OF_TURN))
    special = set(tokenizer.all_special_ids)
    vocab, sources = plan_vocab(counts, source_ids, special, args.min_count, args.align)

    keep = set(vocab)
    needs_target = bool(sources["not_in_source"]) or any(int(source_ids[i]) in keep for i in stale)
    target_head = None
    if needs_target:
        target_head = read_target_head(args.target_model)
    new_state, fresh = shrink_state(state, source_ids, stale, vocab, target_head)
    vocab_maps(new_state["d2t"], verifier_vocab)   # the loader's checks

    report = {"source_draft": str(args.draft), "draft_vocab_patch": str(args.draft_vocab) if args.draft_vocab else None,
              "target_model": str(args.target_model), "train_jsonl": str(args.train_jsonl),
              "min_count": args.min_count, "align": args.align,
              "source_vocab": int(source_ids.numel()), "new_vocab": len(vocab),
              "fraction_of_source": len(vocab) / source_ids.numel(),
              "from_answers": len(sources["answer"]), "special_added": len(sources["special"]),
              "padding": len(sources["padding"]), "rows_from_target_head": len(fresh),
              "train": coverage(args.train_jsonl, tokenizer, keep, args.block_size)}
    if args.val_jsonl:
        report["val"] = coverage(args.val_jsonl, tokenizer, keep, args.block_size)
    report["rows_from_target_head_tokens"] = [tokenizer.convert_ids_to_tokens(t) for t in fresh]
    write_draft_dir(args.draft, args.output, config, new_state, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows_from_target_head_tokens"}, indent=2), flush=True)
    print(f"wrote {args.output} (use as --init-draft without --draft-vocab)", flush=True)


if __name__ == "__main__":
    main()
