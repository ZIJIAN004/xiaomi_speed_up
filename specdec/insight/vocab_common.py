"""Shared pieces of the block-routed draft vocabulary study.

* the draft vocabulary of a DSpark checkpoint (+ optional KIE vocab patch):
  draft id -> target id, and the inverse lookup;
* the draft ``lm_head`` rows as the trained draft serves them;
* answer token sequences (compact JSON + ``<|im_end|>``, as build_kie_draft_vocab.py
  tokenises them), mapped to draft ids (-1 = not in the draft vocabulary);
* the grouping file every grouping method (A, B, D) writes and coverage reads.

A grouping assigns every draft id either to the always-on shared set
(``token_group == SHARED``) or to exactly one of ``num_groups`` groups.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dspark"))
from idspace_minicpm_dataset import serialize_target  # noqa: E402

END_OF_TURN = "<|im_end|>"
GROUPS_FORMAT = "dspark_vocab_groups_v1"
VOCAB_PATCH_FORMAT = "minicpm_dspark_vocab_patch_v1"   # minicpm_dspark.VOCAB_PATCH_FORMAT
SHARED = -1   # token_group value of the always-on shared set
OOV = -1      # draft id of a target token outside the draft vocabulary


def _safetensors_files(directory: Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {directory}")
    return files


def read_tensor(directory: Path, name: str) -> torch.Tensor:
    """One tensor from a safetensors directory, without loading the others."""
    from safetensors import safe_open

    for file in _safetensors_files(directory):
        with safe_open(str(file), framework="pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"{name} not in {directory}")


def find_tensor_name(directory: Path, suffixes: tuple[str, ...]) -> str:
    """The unique tensor whose name ends with the first matching suffix."""
    from safetensors import safe_open

    names: list[str] = []
    for file in _safetensors_files(directory):
        with safe_open(str(file), framework="pt") as handle:
            names.extend(handle.keys())
    for suffix in suffixes:
        hits = [n for n in names if n == suffix or n.endswith("." + suffix)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"several tensors end with {suffix} in {directory}: {hits}")
    raise KeyError(f"no tensor ending with any of {suffixes} in {directory}")


def target_ids_from_d2t(d2t: torch.Tensor, verifier_vocab: int) -> torch.Tensor:
    """Delta-encoded d2t (target = i + d2t[i]) -> target id of each draft id (checked)."""
    target_ids = torch.arange(d2t.numel(), dtype=torch.long) + d2t.long().cpu()
    if target_ids.min() < 0 or target_ids.max() >= verifier_vocab:
        raise ValueError("d2t maps outside the verifier vocabulary")
    if target_ids.unique().numel() != target_ids.numel():
        raise ValueError("d2t maps two draft ids to one target id")
    return target_ids


def load_draft_vocab(official_draft: Path, vocab_patch: Path | None) -> dict[str, Any]:
    """Draft vocabulary of the official checkpoint, re-pointed by a KIE patch if given."""
    config = json.loads((Path(official_draft) / "config.json").read_text(encoding="utf-8"))
    verifier_vocab = int(config["vocab_size"])
    base_d2t = read_tensor(official_draft, "d2t")
    replaced = torch.zeros(0, dtype=torch.long)
    d2t = base_d2t
    if vocab_patch is not None:
        patch = torch.load(vocab_patch, map_location="cpu", weights_only=True)
        if patch.get("format") != VOCAB_PATCH_FORMAT:
            raise ValueError(f"{vocab_patch}: not a {VOCAB_PATCH_FORMAT} file")
        if not torch.equal(patch["base_d2t"].long(), base_d2t.long()):
            raise ValueError(f"{vocab_patch} was built from another draft vocabulary than {official_draft}")
        d2t = patch["d2t"]
        replaced = patch["replaced_draft_ids"].long()
    return {"target_ids": target_ids_from_d2t(d2t, verifier_vocab), "replaced": replaced,
            "verifier_vocab": verifier_vocab}


def inverse_lookup(target_ids: torch.Tensor, verifier_vocab: int) -> np.ndarray:
    """[verifier_vocab] draft id of each target id, OOV where the draft has none."""
    inverse = np.full(verifier_vocab, OOV, dtype=np.int64)
    inverse[target_ids.numpy()] = np.arange(target_ids.numel())
    return inverse


@torch.no_grad()
def load_draft_head(official_draft: Path, vocab: dict[str, Any], target_model: Path,
                    source: str = "checkpoint") -> torch.Tensor:
    """[draft_vocab, hidden] fp32 draft lm_head rows as training leaves them (head frozen).

    ``checkpoint``: official rows; rows re-pointed by the patch take the target head
    row of their new token (minicpm_dspark.attach_verifier). ``verifier``: every row
    from the target head (what --random-init trains with).
    """
    target_ids, replaced = vocab["target_ids"], vocab["replaced"]
    need_target = source == "verifier" or replaced.numel() > 0
    target_head = None
    if need_target:
        name = find_tensor_name(target_model, ("lm_head.weight", "language_model.embed_tokens.weight",
                                               "embed_tokens.weight"))
        target_head = read_tensor(target_model, name)
        if target_head.shape[0] != vocab["verifier_vocab"]:
            raise ValueError(f"target head {name} has {target_head.shape[0]} rows, "
                             f"draft config says {vocab['verifier_vocab']}")
    if source == "verifier":
        return target_head.index_select(0, target_ids).float()
    if source != "checkpoint":
        raise ValueError(f"unknown head source {source!r}")
    head = read_tensor(official_draft, "lm_head.weight").float()
    if head.shape[0] != target_ids.numel():
        raise ValueError(f"official lm_head has {head.shape[0]} rows, draft vocab has {target_ids.numel()}")
    if replaced.numel():
        head[replaced] = target_head.index_select(0, target_ids[replaced]).float()
    return head


def answer_sequences(jsonl: Path, tokenizer, max_rows: int = 0) -> list[np.ndarray]:
    """Target token ids of every answer: compact JSON then <|im_end|> (build_kie_draft_vocab.py)."""
    eot = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    sequences: list[np.ndarray] = []
    with Path(jsonl).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            answer = serialize_target(json.loads(line)["target"])
            ids = tokenizer(answer, add_special_tokens=False).input_ids + [eot]
            sequences.append(np.asarray(ids, dtype=np.int64))
            if max_rows and len(sequences) >= max_rows:
                break
    if not sequences:
        raise ValueError(f"no answers in {jsonl}")
    return sequences


def to_draft(sequences: list[np.ndarray], inverse: np.ndarray) -> list[np.ndarray]:
    return [inverse[s] for s in sequences]


def draft_counts(draft_sequences: list[np.ndarray], draft_vocab: int) -> np.ndarray:
    """[draft_vocab] occurrences of each draft id (OOV positions ignored)."""
    flat = np.concatenate(draft_sequences)
    return np.bincount(flat[flat >= 0], minlength=draft_vocab)


def save_groups(path: Path, token_group: np.ndarray, num_groups: int, target_ids: torch.Tensor,
                method: str, meta: dict[str, Any]) -> None:
    check_groups(token_group, num_groups)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": GROUPS_FORMAT, "method": method, "num_groups": int(num_groups),
                "token_group": torch.from_numpy(token_group.astype(np.int32)),
                "target_ids": target_ids.clone(), "meta": meta}, path)


def load_groups(path: Path) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != GROUPS_FORMAT:
        raise ValueError(f"{path}: not a {GROUPS_FORMAT} file")
    data["token_group"] = data["token_group"].numpy().astype(np.int64)
    check_groups(data["token_group"], data["num_groups"])
    return data


def check_groups(token_group: np.ndarray, num_groups: int) -> None:
    if token_group.ndim != 1:
        raise ValueError("token_group must be 1-D (one entry per draft id)")
    if token_group.min() < SHARED or token_group.max() >= num_groups:
        raise ValueError(f"token_group values must lie in [{SHARED}, {num_groups})")
    used = np.unique(token_group[token_group >= 0])
    if used.size != num_groups:
        raise ValueError(f"{num_groups - used.size} of {num_groups} groups are empty")
