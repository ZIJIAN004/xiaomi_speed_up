#!/usr/bin/env python3
"""Standard ground-truth Xiaomi-EAGLE training skeleton.

This implements the official data/loss contract but intentionally stops at the
validated target-interface boundary. Run only after implementing the pending
Qwen3.5 multimodal embedding/cache adapter in qwen_eagle_draft.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from idspace_gt_dataset import IDSpaceGroundTruthDataset, collate_single
from qwen_eagle_draft import XiaomiEagleDraft


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/home/mr_jiang/models/xiaomi-ocr-0/model"))
    p.add_argument("--processor", type=Path, default=None)
    p.add_argument("--train-jsonl", type=Path, default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared/train.jsonl"))
    p.add_argument("--image-root", type=Path, default=Path("/home/mr_jiang/qkr/runs/idspace_e3/prepared"))
    p.add_argument("--output-dir", type=Path, default=Path("Eagle3/decoding_SFT/checkpoints"))
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=7)
    p.add_argument("--rollout-decay", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def freeze_target(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def standard_distillation_loss(target_logits, draft_logits, assistant_mask):
    """Full-vocabulary soft CE; mask only assistant next-token positions."""
    target_probs = torch.softmax(target_logits.float().detach(), dim=-1)
    draft_log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
    token_loss = -(target_probs * draft_log_probs).sum(dim=-1)
    mask = assistant_mask.to(device=token_loss.device, dtype=token_loss.dtype)
    # Causal alignment: logits[t] predicts input_ids[t+1].
    mask = mask[..., 1:]
    token_loss = token_loss[..., :-1]
    denominator = mask.sum().clamp_min(1.0)
    return (token_loss * mask).sum() / denominator


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor_path = args.processor or args.model
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=args.trust_remote_code
    )
    target = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, torch_dtype="auto"
    ).to(device)
    freeze_target(target)

    hidden_size = target.config.text_config.hidden_size if hasattr(target.config, "text_config") else target.config.hidden_size
    vocab_size = target.config.text_config.vocab_size if hasattr(target.config, "text_config") else target.config.vocab_size
    intermediate_size = target.config.text_config.intermediate_size if hasattr(target.config, "text_config") else target.config.intermediate_size

    dataset = IDSpaceGroundTruthDataset(args.train_jsonl, args.image_root, processor, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)
    draft = XiaomiEagleDraft(hidden_size, vocab_size, intermediate_size).to(device)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.learning_rate)

    # The target multimodal embedding/feature-to-draft-cache adapter is not
    # guessed here. It must be implemented and validated before training.
    raise NotImplementedError(
        "Pending: build target multimodal embeddings, initialize draft cache, "
        "and run full-sequence 7-step rollout with validated Qwen3.5 adapter"
    )


if __name__ == "__main__":
    main()
