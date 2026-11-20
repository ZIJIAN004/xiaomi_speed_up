#!/usr/bin/env python3
r"""Validate Xiaomi-OCR-0 multimodal embedding construction and cache reuse.

This is a read-only validation script. It does not train, generate, or modify
model weights. It checks the exact Qwen3.5 multimodal path used by the model:
text embedding lookup -> vision features -> placeholder masked_scatter ->
language model forward, then checks that a text token can be appended through
Transformers' DynamicCache after the multimodal prefix.

Run on the remote host from the repository root:
    CUDA_VISIBLE_DEVICES=0 python \
      Eagle3/decoding_SFT/validate_xiaomi_multimodal_cache.py \
      --max-samples 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

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


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/home/mr_jiang/models/xiaomi-ocr-0/model"))
    p.add_argument("--processor", type=Path, default=None)
    p.add_argument("--input-jsonl", type=Path,
                   default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl"))
    p.add_argument("--image-root", type=Path,
                   default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared"))
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "multimodal_cache_validation.json")
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def resolve(row, root: Path) -> Path:
    path = Path(row["image"])
    candidates = [path] if path.is_absolute() else [root / path]
    if not path.is_absolute() and path.parts and path.parts[0] == "images":
        candidates.append(root.parent / path)
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("; ".join(str(x.resolve()) for x in candidates))


def image(path: Path):
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"multipage/animated image: {path}")
        value = source.convert("RGB")
        value.load()
    return value


def json_answer(target: Any) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def shape(x):
    return list(x.shape) if hasattr(x, "shape") else None


def max_abs(x):
    return float(x.detach().float().abs().max().item())


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    a = args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rows = [json.loads(x) for x in a.input_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[:a.max_samples] if a.max_samples > 0 else rows
    if not rows:
        raise ValueError("no rows selected")

    processor_path = a.processor or a.model
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=a.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, trust_remote_code=a.trust_remote_code, torch_dtype="auto"
    ).to("cuda:0").eval()
    base = model.model
    language_model = base.language_model
    embedding = base.get_input_embeddings()

    results = []
    for index, row in enumerate(rows):
        started = time.perf_counter()
        img = image(resolve(row, a.image_root))
        answer = json_answer(row["target"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": PROMPT},
        ]}, {"role": "assistant", "content": answer}]
        template = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        inputs = processor(text=[template], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

        # Native multimodal forward. This is the reference path.
        with torch.inference_mode():
            native = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)

            # Reproduce Qwen3.5's multimodal embedding construction from the
            # modeling code: text lookup, vision features, placeholder mask,
            # masked_scatter, then language_model.
            input_ids = inputs["input_ids"]
            input_embeds = embedding(input_ids)
            image_features = base.get_image_features(
                inputs["pixel_values"], inputs.get("image_grid_thw"), return_dict=True
            )
            image_embeds = torch.cat(image_features.pooler_output, dim=0).to(
                input_embeds.device, input_embeds.dtype
            )
            image_mask, _ = base.get_placeholder_mask(
                input_ids, input_embeds=input_embeds, image_features=image_embeds
            )
            rebuilt_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
            position_ids = base.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                inputs_embeds=rebuilt_embeds,
                attention_mask=inputs.get("attention_mask"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                cache=None,
            )
            manual = language_model(
                inputs_embeds=rebuilt_embeds,
                position_ids=position_ids,
                attention_mask=inputs.get("attention_mask"),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

        native_hidden = native.hidden_states[-1]
        manual_hidden = manual.last_hidden_state
        hidden_diff = max_abs(native_hidden - manual_hidden)
        visual_count = int(image_mask.sum().item())

        # Cache reuse check: prefill the same multimodal prefix through the
        # language model, then append one text embedding with the returned
        # DynamicCache. This validates the cache interface, not draft quality.
        prefix_len = input_ids.shape[1] - 1
        prefix_embeds = rebuilt_embeds[:, :prefix_len]
        prefix_pos = position_ids[:, :, :prefix_len]
        prefix_mask = inputs.get("attention_mask")[:, :prefix_len]
        with torch.inference_mode():
            prefix = language_model(
                inputs_embeds=prefix_embeds,
                position_ids=prefix_pos,
                attention_mask=prefix_mask,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            append = language_model(
                inputs_embeds=rebuilt_embeds[:, prefix_len:prefix_len + 1],
                position_ids=position_ids[:, :, prefix_len:prefix_len + 1],
                attention_mask=inputs.get("attention_mask"),
                cache=prefix.cache,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
        cache_len = prefix.cache.get_seq_length()
        append_shape = shape(append.last_hidden_state)
        append_finite = bool(torch.isfinite(append.last_hidden_state).all().item())
        record = {
            "index": index,
            "status": "ok" if hidden_diff < 5e-3 and append_finite else "failed",
            "input_shape": shape(input_ids),
            "rebuilt_embedding_shape": shape(rebuilt_embeds),
            "image_embedding_shape": shape(image_embeds),
            "visual_placeholder_count": visual_count,
            "native_hidden_shape": shape(native_hidden),
            "manual_language_hidden_shape": shape(manual_hidden),
            "native_manual_hidden_max_abs_diff": hidden_diff,
            "prefix_length": prefix_len,
            "cache_length_after_prefix": int(cache_len),
            "append_hidden_shape": append_shape,
            "append_hidden_finite": append_finite,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        results.append(record)
        img.close()

    summary = {
        "status": "complete" if all(x["status"] == "ok" for x in results) else "failed",
        "model": str(a.model),
        "device": "cuda:0",
        "checks": {
            "native_vs_manual_multimodal_embedding_path": "max absolute hidden-state difference < 5e-3",
            "cache_append": "prefix cache accepts one appended text embedding and returns finite hidden state",
        },
        "results": results,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
