#!/usr/bin/env python3
"""Train the standard ground-truth Xiaomi-EAGLE draft head (batch size 1)."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from torch.utils.data import DataLoader

from idspace_gt_dataset import IDSpaceGroundTruthDataset, collate_single
from qwen_eagle_draft import XiaomiEagleDraft


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--processor", type=Path, default=None)
    p.add_argument("--train-jsonl", type=Path, default=None)
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--draft-vocab", type=Path, default=None,
                   help="draft_vocab.pt from build_draft_vocab.py; omit for full target vocabulary")
    p.add_argument("--toy", action="store_true", help="run a tiny synthetic forward/backward test without loading Xiaomi")
    p.add_argument("--toy-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=7)
    p.add_argument("--rollout-decay", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=1)
    # Defaults follow official EAGLE-3 traineagle3/ds_config.json: peak lr 5e-5,
    # WarmupDecayLR (log warmup, linear decay to 0), AdamW betas (0.9, 0.95),
    # weight decay 0, gradient clipping 0.5, gradient accumulation 2.
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.015,
                   help="warmup fraction of optimizer steps (official 12000/800000)")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--low-layer", type=int, default=1)
    p.add_argument("--mid-layer", type=int, default=12)
    p.add_argument("--high-layer", type=int, default=24)
    p.add_argument(
        "--top-k", type=int, nargs="?", const=3, default=0,
        help="enable cumulative target top-k loss gate; bare flag uses k=3",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--wandb-project", default="xiaomi-eagle-idspace")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-log-every", type=int, default=1)
    return p.parse_args()


def warmup_decay_factor(step, warmup_steps, total_steps):
    """DeepSpeed WarmupDecayLR with its default log warmup, min lr 0.

    ``step`` counts completed optimizer steps (LambdaLR convention).
    """
    if step < warmup_steps:
        return math.log(step + 1) / math.log(warmup_steps)
    return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def target_forward_with_context(target, batch):
    """Capture native post-vision embeddings and positions in the SAME forward.

    No second vision pass, placeholder re-embedding, or inferred M-RoPE formula.
    The pre-hook observes exactly what the loaded target gives its text model.
    """
    captured = {}
    def capture(module, args, kwargs):
        if args:
            raise RuntimeError("expected keyword arguments at native language_model boundary")
        for name in ("inputs_embeds", "position_ids"):
            value = kwargs.get(name)
            if value is None:
                raise RuntimeError(f"native target did not supply {name}; inspect loaded model implementation")
            captured[name] = value.detach().clone()
    handle = target.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        output = target(**{k: v for k, v in batch.items()
                           if k not in {"assistant_mask", "sample_index", "target", "image_path"}},
                        output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("native language_model capture hook was not called")
    return output, captured["inputs_embeds"], captured["position_ids"]


def shift_left(tensor, pad=None):
    """Shift the complete sequence axis (dimension 1), retaining length.

    Visual embeddings move with their sequence slots, just like every other
    embedding. Keeping visual slots fixed would break the official TTT alignment.
    """
    tail = torch.zeros_like(tensor[:, :1]) if pad is None else pad
    return torch.cat((tensor[:, 1:], tail), dim=1)


class TeacherTargets(NamedTuple):
    logits: torch.Tensor            # [B, T, draft_vocab], target logits restricted to draft_ids
    in_vocab: torch.Tensor          # [B, T] bool, t2d[argmax of full target logits]
    topk_ids: Optional[torch.Tensor]  # [B, T, k] target-space ids, None when gating is off


@torch.no_grad()
def prepare_teacher(target_logits, draft_ids, t2d, top_k):
    """Reduce full-vocab target logits ONCE, before the rollout.

    Official EAGLE-3: teacher = softmax(target_logits[..., t2d]) and positions
    whose full-vocab teacher argmax is outside the draft vocab are masked.
    The caller can then free the [B, T, V] tensor.
    """
    target_logits = target_logits.detach()
    if top_k > target_logits.shape[-1]:
        raise ValueError("--top-k exceeds vocabulary size")
    topk_ids = target_logits.topk(top_k, dim=-1).indices if top_k > 0 else None
    in_vocab = t2d.to(target_logits.device)[target_logits.argmax(dim=-1)]
    logits = target_logits.index_select(-1, draft_ids.to(target_logits.device))
    return TeacherTargets(logits, in_vocab, topk_ids)


def rollout_loss(draft, feature, embeddings, teacher, assistant_mask,
                 position_ids, steps, decay, pad_embedding):
    """Official TTT alignment: h[t] + e[t+r+1] predicts token[t+r+2]."""
    current_embeds = shift_left(embeddings, pad_embedding)
    # Teacher index t+1 predicts token[t+2]; logits, in-vocab flags and top-k
    # ids move together. The assistant mask is on the predicted token.
    teacher_logits = shift_left(teacher.logits)
    in_vocab = shift_left(teacher.in_vocab)
    topk_ids = None if teacher.topk_ids is None else shift_left(teacher.topk_ids)
    mask = shift_left(shift_left(assistant_mask))
    active = torch.ones_like(mask, dtype=torch.bool)
    cache = None
    total = feature.new_zeros((), dtype=torch.float32)
    for rollout in range(steps):
        feature, draft_logits, cache = draft.step(
            feature, current_embeds, cache, position_ids=position_ids)
        loss_mask = mask.to(torch.bool) & in_vocab
        total = total + decay ** rollout * distill_loss(teacher_logits, draft_logits, loss_mask, active)
        # Current loss is retained; only subsequent rounds are gated out.
        active = update_topk_active(topk_ids, draft_logits, mask, active, draft.draft_ids)
        current_embeds = shift_left(current_embeds, pad_embedding)
        teacher_logits = shift_left(teacher_logits)
        in_vocab = shift_left(in_vocab)
        if topk_ids is not None:
            topk_ids = shift_left(topk_ids)
        mask = shift_left(mask)
    return total, feature, draft_logits, cache, active


def distill_loss(target_logits, draft_logits, mask, active=None):
    teacher = torch.softmax(target_logits.float().detach(), dim=-1)
    student = torch.log_softmax(draft_logits.float(), dim=-1)
    token_loss = -(teacher * student).sum(dim=-1)
    mask = mask.to(token_loss.device, token_loss.dtype)
    if active is not None:
        mask = mask * active.to(mask.device, mask.dtype)
    return (token_loss * mask).mean()  # official EAGLE reduction over B x T


def update_topk_active(target_topk_ids, draft_logits, assistant_mask, active, draft_ids):
    """Gate on the draft's proposal, compared in TARGET token-id space."""
    if target_topk_ids is None:
        return active
    draft_token = draft_ids.to(draft_logits.device)[draft_logits.detach().argmax(dim=-1)]
    reliable = (target_topk_ids == draft_token.unsqueeze(-1)).any(dim=-1)
    valid = assistant_mask.to(reliable.device, torch.bool)
    return active & (~valid | reliable)


def run_toy(args):
    device = torch.device(args.toy_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--toy-device cuda requested but CUDA is unavailable")
    torch.manual_seed(7)
    hidden, vocab, intermediate = 32, 128, 64
    from toy_rotary import ToyRotary
    rotary = ToyRotary()
    draft_ids = torch.randperm(vocab)[:vocab // 2].sort().values  # exercise the reduced-vocab path
    draft = XiaomiEagleDraft(
        hidden, vocab, intermediate, 4, 2, 8, 1e-6,
        low_layer=1, mid_layer=3, high_layer=5, rollout_steps=3, rotary_emb=rotary,
        draft_ids=draft_ids,
    ).to(device)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    batch, seq = 1, 8
    hidden_states = tuple(torch.randn(batch, seq, hidden, device=device) for _ in range(6))
    feature = draft.initial_feature(hidden_states)
    embeddings = torch.randn(batch, seq, hidden, device=device)
    target_logits = torch.randn(batch, seq, vocab, device=device)
    assistant_mask = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    assistant_mask[:, 3:] = True
    positions = torch.arange(seq, device=device).view(1, 1, -1).expand(3, batch, -1)
    teacher = prepare_teacher(target_logits, draft.draft_ids, draft.t2d, args.top_k)
    total, feature, draft_logits, cache, active = rollout_loss(
        draft, feature, embeddings, teacher, assistant_mask,
        positions, 3, 0.8, torch.zeros_like(embeddings[:, :1]))
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_params = [name for name, p in draft.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
    if len(grad_params) != len(list(draft.named_parameters())):
        raise RuntimeError("toy backward did not produce finite gradients for every draft parameter")
    optimizer.step()
    print(json.dumps({
        "status": "ok", "device": str(device), "loss": float(total.item()),
        "feature_shape": list(feature.shape), "draft_logits_shape": list(draft_logits.shape),
        "cache_rollouts": len(cache), "active_positions": int(active.sum()),
        "draft_vocab_size": int(draft.draft_lm_head.out_features),
        "draft_parameters_with_finite_grad": len(grad_params),
        "top_k": args.top_k,
    }, ensure_ascii=False))


def main():
    args = parse_args()
    if args.rollout_steps < 1 or args.epochs < 1 or args.wandb_log_every < 1 or args.top_k < 0:
        raise ValueError("rollout-steps, epochs, wandb-log-every must be positive; top-k must be nonnegative")
    if args.grad_accum < 1 or args.grad_clip <= 0 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("grad-accum >= 1, grad-clip > 0 and warmup-ratio in [0, 1) are required")
    if args.toy:
        run_toy(args)
        return
    required = {"model": args.model, "train_jsonl": args.train_jsonl, "image_root": args.image_root, "output_dir": args.output_dir}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing required training arguments: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.train_jsonl.is_file():
        raise FileNotFoundError(args.train_jsonl)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
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
    draft_ids = None
    if args.draft_vocab is not None:
        if not args.draft_vocab.is_file():
            raise FileNotFoundError(args.draft_vocab)
        draft_ids = torch.load(args.draft_vocab, map_location="cpu")["draft_ids"]
    draft = XiaomiEagleDraft(
        hidden, vocab, intermediate, heads, kv_heads, head_dim, eps,
        args.low_layer, args.mid_layer, args.high_layer, args.rollout_steps,
        rotary_emb=target.model.language_model.rotary_emb, draft_ids=draft_ids,
    ).to(device)  # FP32 master weights; compute runs under autocast below
    draft_vocab = draft.draft_lm_head.out_features
    target_dtype = next(target.parameters()).dtype
    if target_dtype == torch.bfloat16:
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif target_dtype == torch.float32:
        autocast = contextlib.nullcontext
    else:
        raise NotImplementedError(f"target dtype {target_dtype} needs a GradScaler; load in bfloat16")
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=0.0)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = max(2, round(total_updates * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_decay_factor(s, warmup_updates, total_updates))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb as _wandb
        except ImportError as exc:
            raise RuntimeError("wandb is required; install it in the active environment or use --wandb-mode disabled") from exc
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "model": str(args.model), "train_jsonl": str(args.train_jsonl),
                "max_samples": args.max_samples, "epochs": args.epochs,
                "rollout_steps": args.rollout_steps, "rollout_decay": args.rollout_decay,
                "learning_rate": args.learning_rate, "warmup_ratio": args.warmup_ratio,
                "warmup_updates": warmup_updates, "total_updates": total_updates,
                "grad_accum": args.grad_accum, "grad_clip": args.grad_clip,
                "adam_betas": [0.9, 0.95], "weight_decay": 0.0,
                "draft_param_dtype": "float32", "compute_dtype": str(target_dtype),
                "low_layer": args.low_layer,
                "mid_layer": args.mid_layer, "high_layer": args.high_layer,
                "top_k": args.top_k, "hidden_size": hidden, "vocab_size": vocab,
                "draft_vocab": str(args.draft_vocab), "draft_vocab_size": draft_vocab,
                "device": str(device), "dataset_size": len(dataset),
            },
        )

    total_steps = args.epochs * len(loader)
    global_step = 0
    optimizer_step = 0
    grad_norm = float("nan")
    run_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        draft.train()
        for step, raw in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
            # The last group of an epoch may be shorter; average over its real size.
            group_start = step - step % args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            update_now = step - group_start + 1 == group_size
            capacity = getattr(text_config, "max_position_embeddings", None)
            if capacity and batch["input_ids"].shape[1] + args.rollout_steps - 1 > capacity:
                raise ValueError("expanded multimodal sequence plus rollout exceeds target position capacity")
            with torch.no_grad():
                target_out, multimodal_embeds, position_ids = target_forward_with_context(target, batch)
                target_hidden_states = target_out.hidden_states
                teacher = prepare_teacher(target_out.logits, draft.draft_ids, draft.t2d, args.top_k)
                del target_out  # release the [1, T, V] full-vocab logits before the rollout
                # Official padding shifts token IDs and appends ID 0, then embeds.
                pad_embedding = target.get_input_embeddings()(torch.zeros_like(batch["input_ids"][:, :1]))
                # teacher.in_vocab[t] is about the token at t+1; mask on that token.
                predicted = batch["assistant_mask"][:, 1:].to(torch.bool)
                in_vocab_hits = int((teacher.in_vocab[:, :-1] & predicted).sum().item())
                in_vocab_ratio = in_vocab_hits / max(int(predicted.sum().item()), 1)
            # Fusion is TRAINABLE: it must not be inside the target no_grad block.
            with autocast():
                feature = draft.initial_feature(target_hidden_states)
                del target_hidden_states
                total, feature, draft_logits, cache, active = rollout_loss(
                    draft, feature, multimodal_embeds, teacher, batch["assistant_mask"],
                    position_ids, args.rollout_steps, args.rollout_decay, pad_embedding)
            (total / group_size).backward()
            if update_now:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm=args.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            global_step += 1
            elapsed = time.perf_counter() - run_started
            steps_per_sec = global_step / max(elapsed, 1e-9)
            remaining_steps = max(total_steps - global_step, 0)
            eta_sec = remaining_steps / max(steps_per_sec, 1e-9)
            assistant_tokens = int(batch["assistant_mask"].sum().item())
            active_positions = int(active.sum().item())
            metrics = {
                "train/loss": float(total.item()),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": grad_norm,  # from the latest optimizer update
                "train/epoch": epoch,
                "train/step": step,
                "train/global_step": global_step,
                "train/optimizer_step": optimizer_step,
                "train/rollout_steps": args.rollout_steps,
                "train/assistant_tokens": assistant_tokens,
                "train/teacher_argmax_in_draft_vocab": in_vocab_ratio,
                "train/active_positions_after_gate": active_positions,
                "train/active_ratio_after_gate": active_positions / max(int(active.numel()), 1),
                "perf/step_time_sec": elapsed / global_step,
                "perf/steps_per_sec": steps_per_sec,
                "perf/samples_per_sec": steps_per_sec,
                "perf/processed_samples": global_step,
                "perf/total_samples": total_steps,
                "perf/progress": global_step / max(total_steps, 1),
                "perf/eta_sec": eta_sec,
                "perf/elapsed_sec": elapsed,
                "perf/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                "perf/gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
            }
            if wandb is not None and (global_step % args.wandb_log_every == 0 or global_step == total_steps):
                wandb.log(metrics, step=global_step)
            print(
                f"epoch={epoch} step={step} global_step={global_step}/{total_steps} "
                f"loss={total.item():.6f} samples/s={steps_per_sec:.4f} "
                f"eta={eta_sec/60:.1f}min active={active_positions}/{active.numel()}",
                flush=True,
            )
        torch.save({"draft": draft.state_dict(), "args": vars(args)}, args.output_dir / f"epoch_{epoch}.pt")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
