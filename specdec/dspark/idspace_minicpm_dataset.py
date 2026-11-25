"""IDSpace ground-truth samples rendered the way the fine-tuned MiniCPM-OCR sees them.

Two tasks:

* ``kie`` (default): the identity-card KIE stage the MiniCPM-OCR LoRA was trained
  on (``kie_task.py``). Input = layout-region crop (union of blocks + 3% margin)
  shrunk to 262144 px, ``max_slice_nums=1``, the 16-field prompt; the message
  structure is the one of the LoRA ``train.py`` collate. Rows need an ``index``
  that the layout.jsonl knows.
* ``idspace_legacy``: the earlier setup (full image, serving resize to a 2048
  longest side, the 12-field IDSpace prompt of ``specdec/eagle3``).

Both use the EAGLE-3 IDSpace row loader, image lookup and compact JSON answer, and
both build the same masks:

* ``assistant_mask`` = answer tokens + the closing ``<|im_end|>``;
* ``loss_mask[i]`` marks positions whose NEXT token is supervised. With
  ``sample_from_anchor=True`` slot k of an anchor block at p predicts token p+k+1
  from the target distribution at p+k, and speculators masks that slot with
  ``loss_mask[p+k]``. So it is the assistant mask shifted left by one.

The user turn is image then text, as in the LoRA training data and the official
MiniCPM-OCR vLLM examples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eagle3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from idspace_gt_dataset import (  # noqa: E402
    PROMPT as IDSPACE_PROMPT,
    END_OF_TURN,
    IDSpaceGroundTruthDataset,
    end_of_turn_id,
    find_subsequence,
    serialize_target,
    split_train_val,
)
from kie_task import KIE_MAX_PIXELS, KIE_MAX_SLICE_NUMS, KIE_PROMPT, load_regions, open_image  # noqa: E402

__all__ = ["IDSPACE_PROMPT", "KIE_PROMPT", "IDSpaceMiniCPMDataset", "add_data_args", "collate_single",
           "make_datasets_factory", "serialize_target", "split_train_val"]

SERVING_MAX_SIDE = 2048
TASKS = ("kie", "idspace_legacy")


def serving_resize(image: Image.Image) -> Image.Image:
    if max(image.size) > SERVING_MAX_SIDE:
        image.thumbnail((SERVING_MAX_SIDE, SERVING_MAX_SIDE), Image.Resampling.LANCZOS)
    return image


class IDSpaceMiniCPMDataset(IDSpaceGroundTruthDataset):
    """One teacher-forced sample per row.

    ``regions`` (``{row index: crop box}``) switches on the KIE input: crop, then
    the ``max_pixels`` budget. Without it the full image gets the serving resize.
    ``max_slice_nums`` is passed to the processor when not None.
    """

    def __init__(self, jsonl_path, image_root, processor, prompt: str = KIE_PROMPT,
                 image_first: bool = True, max_samples: int = 0, regions: dict | None = None,
                 max_pixels: int = KIE_MAX_PIXELS, max_slice_nums: int | None = KIE_MAX_SLICE_NUMS):
        super().__init__(jsonl_path, image_root, processor, max_samples)
        self.prompt = prompt
        self.image_first = image_first
        self.regions = regions
        self.max_pixels = max_pixels
        self.max_slice_nums = max_slice_nums
        if regions is not None:
            missing = sum(1 for row in self.rows if row.get("index") not in regions)
            if missing:
                raise KeyError(f"{missing} rows of {jsonl_path} have no layout region (by 'index')")

    def load_image(self, row: dict[str, Any]) -> tuple[Image.Image, Path]:
        image_path = self._image_path(row)
        if self.regions is not None:
            return open_image(image_path, self.max_pixels, self.regions[row["index"]]), image_path
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.load()
        return serving_resize(image), image_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image, image_path = self.load_image(row)
        answer = serialize_target(row["target"])

        # Message structure of the LoRA train.py collate: the image part carries no
        # object (images go to the processor), the assistant content is a text part.
        text_part = {"type": "text", "text": self.prompt}
        content = [{"type": "image"}, text_part] if self.image_first else [text_part, {"type": "image"}]
        messages = [{"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        kwargs = {} if self.max_slice_nums is None else {"max_slice_nums": self.max_slice_nums}
        encoded = self.processor(text=[rendered], images=[image], return_tensors="pt", **kwargs)
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


def add_data_args(p: argparse.ArgumentParser) -> None:
    """Rendering options shared by train_dspark, check_alignment and measure_lengths."""
    p.add_argument("--task", choices=TASKS, default="kie",
                   help="kie: layout-region crop + 16-field prompt, as the MiniCPM-OCR LoRA was trained "
                        "(needs --layout-jsonl); idspace_legacy: full image + 12-field IDSpace prompt")
    p.add_argument("--layout-jsonl", type=Path, default=None,
                   help="layout.jsonl of the LoRA data run (crop regions, keyed by row 'index')")
    p.add_argument("--max-pixels", type=int, default=KIE_MAX_PIXELS, help="kie: pixel budget after the crop")
    p.add_argument("--max-slice-nums", type=int, default=KIE_MAX_SLICE_NUMS,
                   help="kie: processor max_slice_nums (the LoRA used 1; vLLM must serve with the same)")
    p.add_argument("--prompt-file", type=Path, default=None,
                   help="user prompt text overriding the task default; must be the prompt the target is served with")
    p.add_argument("--text-first", action="store_true",
                   help="text before image in the user turn (default: image first, as in training)")


def data_argv(args) -> list[str]:
    """The add_data_args options of ``args`` as argv (to forward them to another parser)."""
    argv = ["--task", args.task, "--max-pixels", str(args.max_pixels), "--max-slice-nums", str(args.max_slice_nums)]
    if args.layout_jsonl:
        argv += ["--layout-jsonl", str(args.layout_jsonl)]
    if args.prompt_file:
        argv += ["--prompt-file", str(args.prompt_file)]
    if args.text_first:
        argv.append("--text-first")
    return argv


def make_datasets_factory(args, processor):
    """Returns ``make(jsonl_path, max_samples=0) -> IDSpaceMiniCPMDataset`` for the options in args.

    The layout file (hundreds of MB) is parsed once and shared by every split.
    """
    if args.task == "kie":
        if args.layout_jsonl is None:
            raise ValueError("--task kie needs --layout-jsonl (the crop regions the target was trained on)")
        regions = load_regions(args.layout_jsonl)
        prompt, max_slice_nums = KIE_PROMPT, args.max_slice_nums
    else:
        regions, prompt, max_slice_nums = None, IDSPACE_PROMPT, None
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    def make(jsonl_path, max_samples: int = 0):
        return IDSpaceMiniCPMDataset(jsonl_path, args.image_root, processor, prompt, not args.text_first,
                                     max_samples, regions=regions, max_pixels=args.max_pixels,
                                     max_slice_nums=max_slice_nums)
    return make
