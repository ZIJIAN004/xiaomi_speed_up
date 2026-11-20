#!/usr/bin/env python3
"""Train the standard ground-truth Xiaomi-EAGLE draft head (batch size 1)."""
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
    p.add_argument("--low-layer", type=int, default=1)
    p.add_argument("--mid-layer", type=int, default=12)
    p.add_argument("--high-layer", type=int, default=24)
    p.add_argument(
        "--top-k", type=int, nargs="?", const=3, default=0,
        help="enable cumulative target top-k loss gate; bare flag uses k=3",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_multimodal_embeddings(target, batch):
    """Reproduce Xiaomi's text lookup + visual masked_scatter path."""
    base = target.model
    input_ids = batch["input_ids"]
    embeds = base.get_input_embeddings()(input_ids)
    if "pixel_values" not in batch:
        return embeds
    image_out = base.get_image_features(
        batch["pixel_values"], batch.get("image_grid_thw"), return_dict=True
    )
    image_embeds = torch.cat(image_out.pooler_output, dim=0).to(embeds.device, embeds.dtype)
    image_mask, _ = base.get_placeholder_mask(
        input_ids, inputs_embeds=embeds, image_features=image_embeds
    )
    return embeds.masked_scatter(image_mask, image_embeds)


def shift_text_embeddings(embeddings, input_ids, mm_token_type_ids):
    """Official left shift for text positions while preserving visual positions."""
    shifted = embeddings.clone()
    if mm_token_type_ids is None:
        return torch.cat([embeddings[:, 1:], embeddings[:, -1:]], dim=1)
    for b in range(embeddings.shape[0]):
        text = (mm_token_type_ids[b] == 0).nonzero(as_tuple=False).flatten()
        if text.numel() > 1:
            shifted[b, text[:-1]] = embeddings[b, text[1:]]
            shifted[b, text[-1]] = embeddings[b, text[-1]]
    return shifted


def distill_loss(target_logits, draft_logits, mask, active=None):
    teacher = torch.softmax(target_logits.float().detach(), dim=-1)
    student = torch.log_softmax(draft_logits.float(), dim=-1)
    token_loss = -(teacher * student).sum(dim=-1)
    mask = mask.to(token_loss.device, token_loss.dtype)[..., 1:]
    token_loss = token_loss[..., :-1]
    if active is not None:
        mask = mask * active.to(mask.device, mask.dtype)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def update_topk_active(target_logits, draft_logits, assistant_mask, active, top_k):
    if top_k <= 0:
        return active
    target_topk = target_logits.detach().topk(top_k, dim=-1).indices
    draft_token = draft_logits.detach().argmax(dim=-1, keepdim=True)
    reliable = (target_topk == draft_token).any(dim=-1)[..., :-1]
    valid = assistant_mask.to(reliable.device, torch.bool)[..., 1:]
    return active & (~valid | reliable)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative")
    device = torch.device("cuda:0")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor or args.model, trust_remote_code=args.trust_remote_code)
    target = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, torch_dtype="auto"
    ).to(device)
    freeze(target)
    text_config = getattr(target.config, "text_config", target.config)
    hidden = text_config.hidden_size
    vocab = text_config.vocab_size
    intermediate = text_config.intermediate_size
    heads = text_config.num_attention_heads
    kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, "head_dim", hidden // heads)
    eps = text_config.rms_norm_eps

    dataset = IDSpaceGroundTruthDataset(args.train_jsonl, args.image_root, processor, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)
    draft = XiaomiEagleDraft(
        hidden, vocab, intermediate, heads, kv_heads, head_dim, eps,
        args.low_layer, args.mid_layer, args.high_layer, args.rollout_steps,
    ).to(device, dtype=next(target.parameters()).dtype)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        draft.train()
        for step, raw in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                target_out = target(**{k: v for k, v in batch.items() if k not in {"assistant_mask", "sample_index", "target", "image_path"}},
                                    output_hidden_states=True, return_dict=True, use_cache=False)
                multimodal_embeds = build_multimodal_embeddings(target, batch)
                feature = draft.initial_feature(target_out.hidden_states)
                target_logits = target_out.logits
            total = torch.zeros((), device=device, dtype=torch.float32)
            current_embeds = multimodal_embeds
            cache = None
            active = torch.ones(
                batch["assistant_mask"].shape[0],
                batch["assistant_mask"].shape[1] - 1,
                device=device,
                dtype=torch.bool,
            )
            for rollout in range(args.rollout_steps):
                feature, draft_logits, cache = draft.step(feature, current_embeds, cache)
                total = total + (args.rollout_decay ** rollout) * distill_loss(
                    target_logits, draft_logits, batch["assistant_mask"], active
                )
                active = update_topk_active(
                    target_logits, draft_logits, batch["assistant_mask"], active, args.top_k
                )
                current_embeds = shift_text_embeddings(
                    current_embeds, batch["input_ids"], batch.get("mm_token_type_ids")
                )
            total.backward()
            optimizer.step()
            print(f"epoch={epoch} step={step} loss={total.item():.6f}", flush=True)
        torch.save({"draft": draft.state_dict(), "args": vars(args)}, args.output_dir / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
