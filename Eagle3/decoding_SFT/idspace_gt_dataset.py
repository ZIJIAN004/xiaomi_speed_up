"""IDSpace ground-truth multimodal dataset for Xiaomi-EAGLE training.

This module deliberately does not run model generation. It constructs the
teacher-forced conversation from the official IDSpace target annotation and
returns processor tensors plus the exact assistant-answer span.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

PROMPT = """Extract identity information from this ID card.

Instructions:
1. Extract only information visible in the image.
2. Do not infer missing information.
3. Preserve original names and identification numbers.
4. Normalize unambiguous dates to YYYY-MM-DD.
5. Return null for missing or unreadable fields.
6. Return only valid JSON.

Required fields:
{
  \"document_type\": null,
  \"issuing_country\": null,
  \"given_name\": null,
  \"surname\": null,
  \"date_of_birth\": null,
  \"gender\": null,
  \"nationality\": null,
  \"personal_identification_number\": null,
  \"document_number\": null,
  \"issue_date\": null,
  \"expiry_date\": null,
  \"place_of_issue\": null
}"""


END_OF_TURN = "<|im_end|>"


def end_of_turn_id(tokenizer: Any) -> int:
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {END_OF_TURN} token")
    return int(token_id)


def serialize_target(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def find_subsequence(sequence: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(sequence):
        return []
    width = len(needle)
    return [
        start for start in range(len(sequence) - width + 1)
        if sequence[start:start + width] == needle
    ]


class IDSpaceGroundTruthDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, image_root: str | Path,
                 processor: Any, max_samples: int = 0):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.processor = processor
        rows = [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = rows[:max_samples] if max_samples else rows
        if not self.rows:
            raise ValueError(f"no IDSpace rows found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: dict[str, Any]) -> Path:
        path = Path(row["image"])
        candidates = [path] if path.is_absolute() else [self.image_root / path]
        if path.parts and path.parts[0] == "images":
            candidates.append(self.image_root.parent / path)
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("; ".join(str(p.resolve()) for p in candidates))

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._image_path(row)
        answer = serialize_target(row["target"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()

        user_messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        full_messages = user_messages + [{"role": "assistant", "content": answer}]
        full_template = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        encoded = self.processor(
            text=[full_template], images=[image], return_tensors="pt",
            padding=False, truncation=False, add_special_tokens=False
        )
        image.close()

        input_ids = encoded["input_ids"][0].tolist()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False).input_ids
        matches = find_subsequence(input_ids, answer_ids)
        if len(matches) != 1:
            raise ValueError(f"sample {index}: ground-truth answer span matches={len(matches)}")
        answer_start = matches[0]
        answer_end = answer_start + len(answer_ids)
        # The end-of-turn token is supervised too, so the draft learns to
        # propose it instead of wasting the final rollout of every answer.
        eot = end_of_turn_id(self.processor.tokenizer)
        if answer_end >= len(input_ids) or input_ids[answer_end] != eot:
            raise ValueError(f"sample {index}: answer is not followed by {END_OF_TURN}")

        assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        assistant_mask[answer_start:answer_end + 1] = True
        # Preserve ALL native processor ranks, including [4,1,T] positions,
        # [N,3] grids and [P,C] pixel patches. The batch-size-one collator is identity.
        if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[0] != 1:
            raise ValueError("processor must return input_ids [1,T]")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("batch-size-one training expects no padded tokens")
        encoded = {key: value for key, value in encoded.items() if torch.is_tensor(value)}
        encoded.update({
            "assistant_mask": assistant_mask.unsqueeze(0),
            "sample_index": index,
            "image_path": str(image_path),
            "target": row["target"],
        })
        return encoded


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one multimodal sample and restore model batch dimensions."""
    if len(batch) != 1:
        raise NotImplementedError("multimodal collation currently supports batch_size=1 only")
    return batch[0]
