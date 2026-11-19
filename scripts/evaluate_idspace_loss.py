#!/usr/bin/env python3
"""Reproduce the IDSpace LoRA repository's teacher-forced loss evaluation.

This intentionally mirrors idspace_e3_train.py / idspace_e3_evaluate.py:
image + extraction prompt + gold JSON are passed through the processor, prompt
and padding labels are masked with -100, and the model's causal-LM loss is
aggregated by valid target-token count. It does not call generate() and does
not compute field accuracy or JSON exact match.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

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


class JsonlDataset(Dataset):
    def __init__(self, path: Path, root: Path, max_samples: int = 0):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = rows[:max_samples] if max_samples else rows
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def build_batch(rows: list[dict[str, Any]], processor, root: Path, max_pixels: int) -> dict[str, torch.Tensor]:
    images = []
    user_texts = []
    full_texts = []
    prompt_lengths = []
    try:
        for row in rows:
            image = Image.open(root / row["image"]).convert("RGB")
            images.append(image)
            target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
            user = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            full = user + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
            user_text = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=False)
            full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
            user_texts.append(user_text)
            full_texts.append(full_text)
            prompt_inputs = processor(
                text=user_text,
                images=[image],
                return_tensors="pt",
                min_pixels=min(65536, max_pixels),
                max_pixels=max_pixels,
            )
            prompt_lengths.append(int(prompt_inputs["input_ids"].shape[1]))

        inputs = processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            min_pixels=min(65536, max_pixels),
            max_pixels=max_pixels,
        )
        labels = inputs["input_ids"].clone()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, :prompt_length] = -100
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs
    finally:
        for image in images:
            image.close()


def load_model(model_path: Path):
    import transformers

    loader = getattr(transformers, "AutoModelForImageTextToText", None)
    if loader is not None:
        try:
            return loader.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
            )
        except (ValueError, OSError, TypeError):
            pass
    return transformers.AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )


def evaluate_split(model, processor, path: Path, root: Path, device: torch.device,
                   batch_size: int, max_pixels: int, max_samples: int) -> dict[str, Any]:
    dataset = JsonlDataset(path, root, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda rows: build_batch(rows, processor, root, max_pixels),
    )
    model.eval()
    loss_sum = 0.0
    valid_tokens = 0
    samples = 0
    batches = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            tokens = int((batch["labels"] != -100).sum().item())
            loss_sum += float(output.loss.detach().item()) * tokens
            valid_tokens += tokens
            samples += int(batch["input_ids"].shape[0])
            batches += 1
    loss = loss_sum / max(valid_tokens, 1)
    return {
        "schema": "idspace_teacher_forced_loss_v1",
        "split": path.stem,
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 80 else None,
        "valid_label_tokens": valid_tokens,
        "samples": samples,
        "batches": batches,
        "seconds": time.perf_counter() - started,
        "evaluation_mode": "teacher_forced_gold_json_token_loss",
        "prompt": PROMPT,
        "label_policy": "assistant JSON tokens only; prompt, visual tokens and padding are -100",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared IDSpace directory containing split JSONL and images/")
    parser.add_argument("--model", type=Path, required=True, help="Base Xiaomi-OCR-0 model directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON summary path")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_pixels < 1 or args.max_samples < 0:
        parser.error("batch-size and max-pixels must be positive; max-samples must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA device requested but CUDA is unavailable")

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
        processor.image_processor.min_pixels = min(65536, args.max_pixels)
    model = load_model(args.model).to(device)

    results = []
    for split in args.splits:
        path = args.manifest / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        results.append(evaluate_split(model, processor, path, args.manifest, device,
                                     args.batch_size, args.max_pixels, args.max_samples))

    payload = {
        "schema": "idspace_teacher_forced_loss_report_v1",
        "model": str(args.model),
        "manifest": str(args.manifest),
        "batch_size": args.batch_size,
        "max_pixels": args.max_pixels,
        "device": str(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
