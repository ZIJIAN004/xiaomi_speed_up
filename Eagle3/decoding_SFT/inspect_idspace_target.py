#!/usr/bin/env python3
r"""Inspect IDSpace ground-truth -> Xiaomi-OCR-0 target-forward alignment.

This script performs no generation and trains no draft head. It loads the local
multimodal checkpoint, builds the assistant answer from IDSpace ``target``,
runs teacher-forced target forward on GPU 0, and reports shape/alignment
failures plus quantitative sample statistics.

Remote Linux example:
    python Eagle3/decoding_SFT/inspect_idspace_target.py \
      --model /home/mr_jiang/models/xiaomi-ocr-0 \
      --input-jsonl /home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl \
      --image-root /home/mr_jiang/qkr/runs/idspace_e3/prepared/images \
      --max-samples 10 \
      --output /home/mr_jiang/qkr/runs/idspace_e3/target_inspection.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("/home/mr_jiang/models/xiaomi-ocr-0/model"))
    p.add_argument("--processor", type=Path, default=None,
                   help="processor/tokenizer directory; defaults to --model")
    p.add_argument("--input-jsonl", type=Path,
                   default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl"))
    p.add_argument("--image-root", type=Path,
                   default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared/images"))
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parent / "target_inspection.json")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--low-layer", type=int, default=1,
                   help="hidden-state index for low-level feature; 0 is embedding output")
    p.add_argument("--mid-layer", type=int, default=None,
                   help="hidden-state index for mid-level feature; defaults to middle decoder layer")
    p.add_argument("--high-layer", type=int, default=-1,
                   help="hidden-state index for high-level feature; -1 is final hidden state")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_rows(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"no rows selected from {path}")
    return rows


def resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    value = row.get("image")
    if not isinstance(value, str):
        raise ValueError("row.image is not a string")
    path = Path(value)
    path = path if path.is_absolute() else image_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError(f"multipage/animated image: {path}")
        image = source.convert("RGB")
        image.load()
    return image


def serialize_target(target: Any) -> str:
    if not isinstance(target, dict):
        raise ValueError("row.target is not an object")
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def make_messages(image: Image.Image, answer: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    full = user + [{"role": "assistant", "content": answer}]
    return user, full


def tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if hasattr(value, "shape") else None


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def finite_fraction(value: Any) -> float | None:
    if not hasattr(value, "isfinite"):
        return None
    import torch
    return float(torch.isfinite(value).float().mean().item())


def inspect_row(row: dict[str, Any], index: int, processor: Any, model: Any,
                tokenizer: Any, image_root: Path, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    result: dict[str, Any] = {
        "index": index,
        "sample_id": f"idspace_train_{index:08d}",
        "image": row.get("image"),
        "status": "failed",
        "failure_stage": None,
        "failure": None,
    }
    image = None
    try:
        image_path = resolve_image(row, image_root)
        result["resolved_image"] = str(image_path)
        answer = serialize_target(row.get("target"))
        result["answer_chars"] = len(answer)
        user_messages, full_messages = make_messages(image := load_image(image_path), answer)

        user_template = processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_template = processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        result["user_template_chars"] = len(user_template)
        result["full_template_chars"] = len(full_template)

        # Tokenize both the prompt-only generation prefix and the complete
        # teacher-forced conversation. Exact prefix equality is the boundary
        # check used to build the assistant target mask.
        user_inputs = processor(
            text=[user_template], images=[image], return_tensors="pt", padding=True
        )
        full_inputs = processor(
            text=[full_template], images=[image], return_tensors="pt", padding=True
        )
        input_ids = full_inputs["input_ids"][0].tolist()
        user_ids = user_inputs["input_ids"][0].tolist()
        prefix_mismatch = first_mismatch(user_ids, input_ids)
        result.update({
            "user_input_ids_length": len(user_ids),
            "full_input_ids_length": len(input_ids),
            "prefix_mismatch_index": prefix_mismatch,
            "prefix_exact": prefix_mismatch is None and len(user_ids) <= len(input_ids),
            "processor_shapes": {key: tensor_shape(value) for key, value in full_inputs.items()},
        })
        if prefix_mismatch is not None or len(user_ids) >= len(input_ids):
            raise ValueError("user-only tokenization is not an exact prefix of full conversation")

        assistant_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        assistant_mask[len(user_ids):] = True
        result["assistant_token_count"] = int(assistant_mask.sum().item())
        result["assistant_start"] = len(user_ids)
        result["assistant_end"] = len(input_ids)

        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in full_inputs.items()
        }
        result["input_device"] = str(model_inputs["input_ids"].device)
        result["failure_stage"] = "target_forward"
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        result["logits_shape"] = tensor_shape(logits)
        result["hidden_state_count"] = len(hidden_states) if hidden_states is not None else 0
        result["hidden_state_shapes"] = [tensor_shape(value) for value in hidden_states] if hidden_states else []
        result["logits_finite_fraction"] = finite_fraction(logits)
        if logits.ndim != 3 or logits.shape[1] != len(input_ids):
            raise ValueError("logits sequence length does not match input_ids")
        if hidden_states is None or len(hidden_states) < 2:
            raise ValueError("target model did not return hidden-state layers")
        if result["full_input_ids_length"] > args.max_model_len:
            raise ValueError("expanded multimodal sequence exceeds configured max-model-len")

        # Verify the embedding lookup needed by the draft head. This is not a
        # second model pass: it is the frozen target embedding table queried by
        # the same teacher-forced input IDs.
        embedding_layer = model.get_input_embeddings()
        token_embeddings = embedding_layer(model_inputs["input_ids"])
        result["token_embeddings_shape"] = tensor_shape(token_embeddings)
        if token_embeddings.shape[:2] != logits.shape[:2]:
            raise ValueError("target token embeddings and logits have different [batch, sequence] shapes")
        result["token_embeddings_finite_fraction"] = finite_fraction(token_embeddings)

        # Select the three feature levels that the future EAGLE fusion layer
        # will consume. The script validates shape compatibility only; it does
        # not choose final layers for the model implementation.
        hidden_count = len(hidden_states)
        mid_index = args.mid_layer if args.mid_layer is not None else hidden_count // 2
        selected_indices = [args.low_layer, mid_index, args.high_layer]
        selected_indices = [index if index >= 0 else hidden_count + index for index in selected_indices]
        if any(index < 0 or index >= hidden_count for index in selected_indices):
            raise ValueError(f"selected hidden layer is out of range: {selected_indices}, count={hidden_count}")
        selected = [hidden_states[index] for index in selected_indices]
        selected_shapes = [tensor_shape(value) for value in selected]
        result.update({
            "selected_feature_layers": selected_indices,
            "selected_feature_shapes": selected_shapes,
            "selected_feature_hidden_size": [shape[-1] for shape in selected_shapes],
        })
        if any(value.shape != selected[0].shape for value in selected[1:]):
            raise ValueError("selected low/mid/high hidden states do not have identical shapes")
        fused_input = torch.cat(selected, dim=-1)
        result["feature_fusion_input_shape"] = tensor_shape(fused_input)
        result["feature_fusion_output_shape_if_linear_3H_to_H"] = [
            fused_input.shape[0], fused_input.shape[1], selected[0].shape[-1]
        ]
        if finite_fraction(fused_input) != 1.0:
            raise ValueError("selected hidden states contain NaN or Inf")

        # Causal alignment: logits[:, t] predicts input_ids[t+1]. Only target
        # positions whose target token is in the assistant span are counted.
        prediction = logits[0, :-1].argmax(dim=-1).cpu()
        next_ids = torch.tensor(input_ids[1:], dtype=prediction.dtype)
        valid = assistant_mask[1:]
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            raise ValueError("assistant prediction mask is empty after causal shift")
        next_token_accuracy = float((prediction[valid] == next_ids[valid]).float().mean().item())

        # Validate the exact teacher loss operation needed later. This is a
        # self-teacher numerical check, not a quality score.
        assistant_logits = logits[0, :-1][valid]
        teacher_probs = torch.softmax(assistant_logits.float(), dim=-1)
        teacher_log_probs = torch.log_softmax(assistant_logits.float(), dim=-1)
        teacher_ce = -(teacher_probs * teacher_log_probs).sum(dim=-1).mean()
        probability_sum_error = float((teacher_probs.sum(dim=-1) - 1).abs().max().item())
        result.update({
            "causal_logit_positions": int(logits.shape[1] - 1),
            "assistant_prediction_positions": valid_count,
            "assistant_next_token_argmax_accuracy": next_token_accuracy,
            "assistant_mask_shift": "assistant_mask[1:] aligns logits[:, :-1] with input_ids[1:]",
            "teacher_probability_sum_max_error": probability_sum_error,
            "teacher_self_cross_entropy": float(teacher_ce.item()),
            "teacher_loss_finite": bool(torch.isfinite(teacher_ce).item()),
            "status": "ok",
            "failure_stage": None,
        })
        if probability_sum_error > 1e-4 or not torch.isfinite(teacher_ce):
            raise ValueError("teacher distribution/loss is numerically invalid")
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if image is not None:
            image.close()
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def main() -> int:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    processor_path = args.processor or args.model
    if not processor_path.is_dir():
        raise FileNotFoundError(processor_path)
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(args.input_jsonl)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this validation is required to run on GPU 0")
    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
    )
    model.eval()
    tokenizer = processor.tokenizer
    rows = load_rows(args.input_jsonl, args.max_samples)

    records = []
    for index, row in enumerate(rows):
        record = inspect_row(row, index, processor, model, tokenizer, args.image_root, device, args)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    counts = Counter(record["status"] for record in records)
    failure_stages = Counter(record["failure_stage"] or "none" for record in records if record["status"] != "ok")
    ok = [record for record in records if record["status"] == "ok"]
    summary = {
        "status": "complete" if len(ok) == len(records) else "partial_failure",
        "model": str(args.model),
        "input_jsonl": str(args.input_jsonl),
        "image_root": str(args.image_root),
        "device": "cuda:0",
        "selected_samples": len(records),
        "status_counts": dict(counts),
        "failure_stage_counts": dict(failure_stages),
        "failure_rate": (len(records) - len(ok)) / len(records),
        "prefix_alignment_failure_rate": sum(
            record.get("prefix_mismatch_index") is not None for record in records
        ) / len(records),
        "mean_assistant_tokens_ok": (
            sum(record["assistant_token_count"] for record in ok) / len(ok) if ok else None
        ),
        "mean_assistant_next_token_argmax_accuracy_ok": (
            sum(record["assistant_next_token_argmax_accuracy"] for record in ok) / len(ok) if ok else None
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not failure_stages else 2


if __name__ == "__main__":
    raise SystemExit(main())
