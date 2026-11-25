"""The identity-card KIE task exactly as the MiniCPM-OCR LoRA was trained and evaluated.

Mirror of ``/home/mr_jiang/Lora/reproduction/minicpm_official_9to1_20261123_230559_273596/common.py``
(FIELDS, PROMPT, open_image, load_regions) and of its ``train.py`` collate:

* input  = the union of all detected layout blocks plus a 3% margin (at least 2 px),
  clipped to the image; a record with no blocks keeps the full image;
* the crop is then shrunk to at most ``max_pixels`` (262144) pixels, LANCZOS;
* the processor runs with ``max_slice_nums=1``;
* user turn = [image, 16-field prompt]; assistant = compact JSON of the target.

The draft must see the same tokens the fine-tuned target sees when served, so any
change here has to match the target's serving code. ``KIE_PROMPT`` is checked
against the LoRA run's ``training/config.json`` on the server (see README).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

FIELDS = (
    "card_access_number", "date_of_birth", "date_of_expiry", "date_of_issue",
    "document_number", "document_type", "gender", "given_name",
    "issuing_authority", "issuing_country", "nationality",
    "personal_identification_number", "place_of_birth", "printed_initials",
    "second_surname", "surname",
)
KIE_PROMPT = (
    'Extract the identity-card information visible in this image. Return ONLY one JSON object. '
    'Include EVERY key in this exact schema, even when its value is null:\n'
    + json.dumps({key: None for key in FIELDS}, separators=(',', ':')) + '\n'
    'Use null when a field is absent or unreadable. Preserve visible spelling, accents, '
    'punctuation and identifier characters. Normalize unambiguous dates to YYYY-MM-DD. '
    'given_name is the first name; surname is the family name. document_number is the '
    'card number, and personal_identification_number is the personal number. '
    'document_type is identity_card. issuing_country uses the ISO three-letter country code. '
    'Do not infer missing information, omit keys, or add commentary.'
)
KIE_MAX_PIXELS = 262144
KIE_MAX_SLICE_NUMS = 1


def region_from_layout(layout: dict) -> list[int]:
    """Union of the detected blocks plus a 3% margin (>= 2 px), clipped; full image if no blocks."""
    width, height = layout["image_width"], layout["image_height"]
    boxes = [b["bbox"] for b in layout["blocks"]]
    if boxes:
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
        margin_x = max(2, int((x2 - x1) * .03)); margin_y = max(2, int((y2 - y1) * .03))
        bbox = [max(0, x1 - margin_x), max(0, y1 - margin_y), min(width, x2 + margin_x), min(height, y2 + margin_y)]
    else:
        bbox = [0, 0, width, height]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise RuntimeError("Invalid official layout region union")
    return bbox


def load_regions(layout_jsonl: str | Path) -> dict:
    """{record index: crop box} for every record of a layout.jsonl."""
    regions = {}
    with Path(layout_jsonl).open(encoding="utf8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            regions[record["index"]] = region_from_layout(record["layout"])
    if not regions:
        raise ValueError(f"no layout records in {layout_jsonl}")
    return regions


def open_image(path: str | Path, max_pixels: int, bbox=None):
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
    if bbox is not None:
        cropped = image.crop(bbox)
        image.close()
        image = cropped
    if image.width * image.height > max_pixels:
        ratio = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                             Image.Resampling.LANCZOS)
    return image
