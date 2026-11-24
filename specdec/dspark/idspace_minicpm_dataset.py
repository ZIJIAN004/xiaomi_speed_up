"""IDSpace ground-truth samples rendered the way MiniCPM-OCR is served.

Reuses the EAGLE-3 IDSpace helpers (row loading, image lookup, seeded split,
answer serialization) so both drafts see the same train/validation rows. Two
things are specific to DSpark:

* ``loss_mask[i]`` marks positions whose NEXT token is supervised (answer tokens
  and the closing ``<|im_end|>``). With ``sample_from_anchor=True`` slot k of an
  anchor block at p predicts token p+k+1 from the target distribution at p+k,
  and speculators masks that slot with ``loss_mask[p+k]``.
* The user turn is image then text, as in the official MiniCPM-OCR vLLM
  examples ("Place the image before the text") and ms-swift fine-tuning data
  ("<image>Extract the ...").
* The image goes through the serving-side resize (longest side capped at 2048,
  LANCZOS; ``baseline/harness/processor_audit.py:40``) before the processor, so
  image token counts match what vLLM sees.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eagle3"))
from idspace_gt_dataset import (  # noqa: E402
    PROMPT as IDSPACE_PROMPT,
    END_OF_TURN,
    IDSpaceGroundTruthDataset,
    end_of_turn_id,
    find_subsequence,
    serialize_target,
    split_train_val,
)

__all__ = ["IDSPACE_PROMPT", "IDSpaceMiniCPMDataset", "collate_single", "split_train_val"]

SERVING_MAX_SIDE = 2048


def serving_resize(image: Image.Image) -> Image.Image:
    if max(image.size) > SERVING_MAX_SIDE:
        image.thumbnail((SERVING_MAX_SIDE, SERVING_MAX_SIDE), Image.Resampling.LANCZOS)
    return image


class IDSpaceMiniCPMDataset(IDSpaceGroundTruthDataset):
    def __init__(self, jsonl_path, image_root, processor, prompt: str = IDSPACE_PROMPT,
                 image_first: bool = True, max_samples: int = 0):
        super().__init__(jsonl_path, image_root, processor, max_samples)
        self.prompt = prompt
        self.image_first = image_first

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self._image_path(row)
        answer = serialize_target(row["target"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()
        image = serving_resize(image)

        text_part = {"type": "text", "text": self.prompt}
        image_part = {"type": "image", "image": image}
        content = [image_part, text_part] if self.image_first else [text_part, image_part]
        messages = [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        encoded = self.processor(text=[rendered], images=[image], return_tensors="pt",
                                 padding=False, truncation=False, add_special_tokens=False)
        image.close()

        if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[0] != 1:
            raise ValueError("processor must return input_ids [1, T]")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("batch-size-one training expects no padded tokens")
        input_ids = encoded["input_ids"][0].tolist()
        answer_ids = self.processor.tokenizer(answer, add_special_tokens=False).input_ids
        matches = find_subsequence(input_ids, answer_ids)
        if len(matches) != 1:
            raise ValueError(f"sample {index}: ground-truth answer span matches={len(matches)}")
        start, end = matches[0], matches[0] + len(answer_ids)
        eot = end_of_turn_id(self.processor.tokenizer)
        if end >= len(input_ids) or input_ids[end] != eot:
            raise ValueError(f"sample {index}: answer is not followed by {END_OF_TURN}")
        if start < 1:
            raise ValueError(f"sample {index}: answer starts at position 0")

        assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        assistant_mask[start:end + 1] = True          # answer + <|im_end|>
        loss_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        loss_mask[start - 1:end] = True               # positions predicting those tokens
        return {
            "inputs": {k: v for k, v in encoded.items() if torch.is_tensor(v)},
            "assistant_mask": assistant_mask.unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0),
            "sample_index": index,
            "image_path": str(image_path),
        }


def collate_single(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise NotImplementedError("multimodal collation supports batch_size=1 only")
    return batch[0]
