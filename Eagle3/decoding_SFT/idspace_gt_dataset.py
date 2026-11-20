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
            text=[full_template], images=[image], return_tensors="pt", padding=True
        )
        image.close()

        input_ids = encoded["input_ids"][0].tolist()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False).input_ids
        matches = find_subsequence(input_ids, answer_ids)
        if len(matches) != 1:
            raise ValueError(f"sample {index}: ground-truth answer span matches={len(matches)}")
        answer_start = matches[0]
        answer_end = answer_start + len(answer_ids)

        assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        assistant_mask[answer_start:answer_end] = True
        # Keep image_grid_thw as [num_images, 3]. Only tensors with a
        # leading batch dimension are unbatched here; Qwen3.5 expects
        # image_grid_thw to remain rank-2 in get_image_features().
        # Keep processor tensors unbatched except image_grid_thw, whose
        # processor batch shape [num_images, 3] is already the model shape.
        # The training loop adds the batch dimension to the rank-1 tensors.
        encoded = {
            key: (value if key == "image_grid_thw" else value.squeeze(0))
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
        encoded.update({
            "assistant_mask": assistant_mask,
            "sample_index": index,
            "image_path": str(image_path),
            "target": row["target"],
        })
        return encoded


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one multimodal sample and restore model batch dimensions."""
    if len(batch) != 1:
        raise NotImplementedError("multimodal collation currently supports batch_size=1 only")
    sample = batch[0]
    batched = {}
    for key, value in sample.items():
        if not torch.is_tensor(value):
            batched[key] = value
        elif key in {"image_grid_thw", "pixel_values"}:
            # Qwen3.5 processor expects image_grid_thw [num_images, 3] and
            # pixel_values [num_patches, channels] for a single image.
            batched[key] = value
        else:
            batched[key] = value.unsqueeze(0)
    return batched
