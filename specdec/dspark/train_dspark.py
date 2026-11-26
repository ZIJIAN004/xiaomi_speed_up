#!/usr/bin/env python3
"""Train a DSpark draft for (fine-tuned) MiniCPM-OCR 1.0 on IDSpace ground truth.

Default task: the identity-card KIE stage (layout-region crop, 16-field prompt,
max_slice_nums=1) exactly as the MiniCPM-OCR LoRA was trained (kie_task.py).

Frozen HF target (multimodal, batch size 1) -> aux hidden states of layers
[3,7,11,15,19] + pre-norm last hidden state -> speculators DSparkDraftModel
(DFlash backbone + Markov head + confidence head, its own loss and metrics),
warm-started from the official MiniCPM-OCR-1.0-DSpark weights. Checkpoints are
written in the official layout, so vLLM serves them with
``--speculative-config '{"method":"dspark","model":"<dir>","num_speculative_tokens":15}'``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch._dynamo  # not inside main(): a local import would make `torch` a local name there
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idspace_minicpm_dataset import add_data_args, collate_single, make_datasets_factory, split_train_val  # noqa: E402
from minicpm_dspark import (  # noqa: E402
    TargetFeatureExtractor, accumulate_metrics, apply_draft_vocab, attach_verifier, configure_trainable,
    draft_inputs, load_official_draft, precast_frozen, reduce_metrics, save_official,
)
from grad_probe import parse_steps, probe_due, probe_scalars, run_probe  # noqa: E402
from tracker import Tracker, add_tracker_args  # noqa: E402

# speculators examples/train/dspark_qwen3_0_6b_sharegpt_online.sh
DEFAULT_LOSS_FN = '{"ce": 0.1, "tv": 0.9}'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="MiniCPM-OCR 1.0 target (the fine-tuned one)")
    p.add_argument("--processor", type=Path, default=None, help="defaults to --model")
    p.add_argument("--init-draft", type=Path, required=True,
                   help="official-layout DSpark draft to start from (openbmb/MiniCPM-OCR-1.0-DSpark)")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    add_data_args(p)
    p.add_argument("--max-samples", type=int, default=0,
                   help="keep only the first N rows of each file (smoke runs)")
    p.add_argument("--train-samples", type=int, default=0,
                   help="train on a seeded random subset of N training rows (0 = all); recorded in "
                        "data_split.json")
    p.add_argument("--draft-vocab", type=Path, default=None,
                   help="vocabulary patch from build_kie_draft_vocab.py (same draft vocab size, some "
                        "official tail tokens replaced by answer tokens the official vocab lacks)")
    p.add_argument("--val-jsonl", type=Path, default=None,
                   help="separate validation file (e.g. IDSpace validation.jsonl, the benchmark set); "
                        "when given, all --train-jsonl rows train and --val-fraction is ignored")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None, help="<output-dir>/last/trainer_state.pt")
    p.add_argument("--eval-only", action="store_true",
                   help="evaluate --init-draft on the validation split and exit (alignment gate)")
    p.add_argument("--eval-limit", type=int, default=0, help="cap on evaluated samples (0 = all)")
    # Optimisation defaults: speculators TrainerConfig (AdamW wd 0.01, linear schedule,
    # clip 1.0) and the DSpark example script (lr 3e-4, ce 0.1 + tv 0.9, alpha 1.0).
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="samples per optimizer step; ~16 IDSpace answers give the ~3k anchors of one "
                        "speculators packed batch (max_anchors 3072)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke runs)")
    p.add_argument("--loss-fn", default=DEFAULT_LOSS_FN)
    p.add_argument("--loss-implementation", choices=("fused", "eager"), default="fused")
    p.add_argument("--confidence-head-alpha", type=float, default=1.0)
    p.add_argument("--decay-gamma", type=float, default=4.0, help="dflash_loss_decay gamma")
    p.add_argument("--max-anchors", type=int, default=256,
                   help="anchor blocks per sample; fixed so the compiled draft keeps one shape")
    p.add_argument("--train-lm-head", action="store_true",
                   help="also train the draft lm_head (speculators keeps it frozen)")
    p.add_argument("--lm-head-init", choices=("checkpoint", "verifier"), default=None,
                   help="draft lm_head from the init checkpoint, or from the target head rows (default: "
                        "checkpoint; verifier with --random-init)")
    p.add_argument("--random-init", action="store_true",
                   help="start from a randomly initialised draft (speculators' own init) with the --init-draft "
                        "architecture and vocabulary instead of its weights; the frozen draft lm_head becomes the "
                        "target head rows, as speculators trains from scratch")
    p.add_argument("--init-seed", type=int, default=None, help="seed of --random-init (default: --seed)")
    p.add_argument("--probe-steps", default="",
                   help="comma-separated optimizer steps at which grad_probe.py measures per-slot gradients "
                        "(0 = before the first update); see grad_probe.py")
    p.add_argument("--probe-every", type=int, default=0,
                   help="after the last --probe-steps entry, probe every N optimizer steps (0 = never)")
    p.add_argument("--probe-samples", type=int, default=64, help="fixed training rows used by every probe")
    p.add_argument("--probe-groups", type=int, default=4,
                   help="disjoint sample groups; cross-group products give the noise-free (signal) view, "
                        "same-group ones the batch view (64/4 = the 16-sample optimizer batch). RAM: groups x "
                        "(block_size + 1) x trainable params x 4 bytes")
    p.add_argument("--probe-step-check", action=argparse.BooleanOptionalAction, default=True,
                   help="also take a small real step per probe to validate the first-order predictions")
    p.add_argument("--stop-after-steps", type=int, default=0,
                   help="stop after N optimizer steps without shortening the LR schedule (unlike --max-steps)")
    p.add_argument("--select-metric", choices=("eal", "accept_len"), default="eal",
                   help="validation metric for best/: eal = greedy (temperature 0, as served), "
                        "accept_len = rejection-sampling expectation")
    p.add_argument("--require-loss-decrease", action=argparse.BooleanOptionalAction, default=True,
                   help="an epoch counts as better only if --select-metric rises AND val loss falls, both "
                        "against the best checkpoint so far (initially the init draft)")
    p.add_argument("--full-target-forward", action="store_true",
                   help="run the full target (with LM head) instead of its base model for features")
    p.add_argument("--profile-stages", action="store_true",
                   help="log ms/sample of target forward, draft forward and draft backward; adds "
                        "three CUDA syncs per sample, so keep it off for real training")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile (debugging; slow)")
    p.add_argument("--log-every", type=int, default=10)
    add_tracker_args(p)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.grad_accum < 1 or args.max_anchors < 1 or args.grad_clip <= 0:
        p.error("epochs, grad-accum, max-anchors must be >= 1 and grad-clip > 0")
    if not 0 <= args.warmup_ratio < 1:
        p.error("warmup-ratio must be in [0, 1)")
    if args.train_samples < 0:
        p.error("train-samples must be >= 0")
    if args.task == "kie" and args.layout_jsonl is None:
        p.error("--task kie needs --layout-jsonl")
    if args.random_init and args.lm_head_init == "checkpoint":
        p.error("--random-init has no trained draft lm_head; use --lm-head-init verifier (the default with it)")
    if args.lm_head_init is None:
        args.lm_head_init = "verifier" if args.random_init else "checkpoint"
    if args.init_seed is None:
        args.init_seed = args.seed
    try:
        args.probe_steps = parse_steps(args.probe_steps)
    except ValueError as error:
        p.error(str(error))
    if args.probe_every < 0 or args.stop_after_steps < 0:
        p.error("probe-every and stop-after-steps must be >= 0")
    if (args.probe_steps or args.probe_every) and (args.probe_groups < 2 or args.probe_samples < 2 * args.probe_groups):
        p.error("probing needs --probe-groups >= 2 and --probe-samples >= 2 * --probe-groups")
    return args


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def resolve_split(n_rows: int, args) -> tuple[list[int], list[int]]:
    """Seeded split shared with specdec/eagle3; recorded and verified on restart."""
    val_fraction = 0.0 if args.val_jsonl else args.val_fraction
    train_idx, val_idx = split_train_val(n_rows, val_fraction, args.seed)
    if args.train_samples and args.train_samples < len(train_idx):
        train_idx = sorted(random.Random(args.seed).sample(train_idx, args.train_samples))
    path = args.output_dir / "data_split.json"
    record = {"train_jsonl": str(args.train_jsonl), "val_jsonl": str(args.val_jsonl),
              "max_samples": args.max_samples, "train_samples": args.train_samples,
              "val_fraction": val_fraction, "seed": args.seed, "task": args.task,
              "layout_jsonl": str(args.layout_jsonl),
              "train_indices": train_idx, "val_indices": val_idx}
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["train_indices"] != train_idx or saved["val_indices"] != val_idx:
            raise ValueError(f"{path} disagrees with this run's split; use a new --output-dir")
    else:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return train_idx, val_idx


def build_datasets(args, processor):
    """(train subset, validation dataset or None) for the IDSpace files in args."""
    make = make_datasets_factory(args, processor)
    dataset = make(args.train_jsonl, args.max_samples)
    train_idx, val_idx = resolve_split(len(dataset), args)
    if args.val_jsonl:
        val = make(args.val_jsonl, args.max_samples)
    else:
        val = Subset(dataset, val_idx) if val_idx else None
    return Subset(dataset, train_idx), val


def cuda_time(enabled: bool = True) -> float:
    """Wall time after all queued CUDA work finished (stage timing, not kernel timing).

    Disabled: returns 0 without synchronizing, so the stage times are simply zero.
    """
    if not enabled:
        return 0.0
    torch.cuda.synchronize()
    return time.perf_counter()


class Pipeline:
    """Loaded target + draft + loss settings; one call = one sample's loss and metrics."""

    def __init__(self, args, device: torch.device):
        from speculators.models.dspark import DSparkDraftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(args.processor or args.model)
        self.target = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
        self.target.eval()
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.draft = load_official_draft(args.init_draft,
                                         random_init_seed=args.init_seed if args.random_init else None)
        if args.draft_vocab is not None:
            apply_draft_vocab(self.draft, args.draft_vocab)
        attach_verifier(self.draft, self.target, args.lm_head_init)
        self.trainable = configure_trainable(self.draft.model, args.train_lm_head)
        precast_frozen(self.draft.model, torch.bfloat16, keep=set(self.draft.key_dtypes))
        self.model = self.draft.model.to(device)
        self.extractor = TargetFeatureExtractor(self.target, self.model.config.aux_hidden_state_layer_ids,
                                                use_base_model=not args.full_target_forward)
        train_kwargs, _ = DSparkDraftModel.get_trainer_kwargs(
            loss_fn=args.loss_fn, loss_implementation=args.loss_implementation,
            dflash_decay_gamma=args.decay_gamma, max_anchors=args.max_anchors,
            confidence_head_alpha=args.confidence_head_alpha)
        self.call_kwargs = train_kwargs
        self.max_anchors = args.max_anchors
        self.profile = args.profile_stages

    def draft_batch(self, raw):
        """Target features + ids + mask of one collated sample, in the layout the draft forward takes."""
        inputs = {k: v.to(self.device) for k, v in raw["inputs"].items()}
        return draft_inputs(self.extractor(inputs), inputs["input_ids"], raw["loss_mask"].to(self.device))

    def sample(self, raw):
        inputs = {k: v.to(self.device) for k, v in raw["inputs"].items()}
        loss_mask = raw["loss_mask"].to(self.device)
        t0 = cuda_time(self.profile)
        features = self.extractor(inputs)
        t1 = cuda_time(self.profile)
        batch = draft_inputs(features, inputs["input_ids"], loss_mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, metrics = self.model(**batch, **self.call_kwargs)
        t2 = cuda_time(self.profile)
        anchors = int(loss_mask[:, :-self.model.block_size].sum())
        return loss, metrics, {"seq_len": inputs["input_ids"].shape[1], "valid_anchors": anchors,
                               "anchors_truncated": anchors > self.max_anchors,
                               "t_target": t1 - t0, "t_draft_fwd": t2 - t1}


@torch.no_grad()
def evaluate(pipe: Pipeline, loader, seed: int, limit: int = 0) -> dict[str, float]:
    was_training = pipe.model.training
    pipe.model.eval()
    sums: dict[str, float] = {}
    n = truncated = 0
    with torch.random.fork_rng(devices=[pipe.device]):
        torch.manual_seed(seed)  # anchor subsampling is random only for over-long answers
        for raw in loader:
            _, metrics, info = pipe.sample(raw)
            accumulate_metrics(sums, metrics)
            n += 1
            truncated += int(info["anchors_truncated"])
            if limit and n >= limit:
                break
    pipe.model.train(was_training)
    out = {f"val/{k}": v for k, v in reduce_metrics(sums).items()}
    out["val/samples"] = n
    out["val/anchors_truncated_frac"] = truncated / max(n, 1)
    return out


def is_improvement(val_metrics: dict[str, float], best: dict[str, float] | None, select: str,
                   require_loss_decrease: bool) -> bool:
    """Better than the best so far: higher ``select`` and, if required, lower val loss."""
    if not val_metrics:
        return False
    if best is None:
        return True
    if not val_metrics[select] > best[select]:
        return False
    return not require_loss_decrease or val_metrics["val/loss"] < best["val/loss"]


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}


def save_trainer_state(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def print_metrics(tag: str, metrics: dict[str, float]) -> None:
    keys = [k for k in metrics if k.split("/")[-1] in
            ("loss", "eal", "accept_len", "accept_rate", "full_acc", "confidence_abs_error", "samples",
             "anchors_truncated_frac")]
    print(f"{tag} " + " ".join(f"{k}={metrics[k]:.4f}" for k in keys), flush=True)
    positions = sorted((int(k.split("/")[-1].split("_")[1]), v) for k, v in metrics.items()
                       if k.split("/")[-1].startswith("position_") and k.endswith("_acc"))
    if positions:
        print(f"{tag} per-position acc " + " ".join(f"{p}:{v:.3f}" for p, v in positions), flush=True)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.model, args.init_draft, args.image_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (args.train_jsonl, args.val_jsonl, args.layout_jsonl, args.draft_vocab):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (flex attention + fused Triton losses)")
    if args.no_compile:
        torch._dynamo.config.disable = True
    device = torch.device("cuda:0")
    generator = set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline(args, device)
    train_set, val_set = build_datasets(args, pipe.processor)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_single,
                            num_workers=2) if val_set is not None else None
    n_trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
    print(f"draft: block_size={pipe.model.block_size} aux_layers={pipe.model.config.aux_hidden_state_layer_ids} "
          f"draft_vocab={pipe.model.draft_vocab_size} trainable={n_trainable/1e6:.1f}M "
          f"train={len(train_set)} val={len(val_set) if val_set is not None else 0}", flush=True)

    if args.eval_only:
        if val_loader is None:
            raise ValueError("--eval-only needs --val-jsonl or --val-fraction > 0")
        metrics = evaluate(pipe, val_loader, args.seed, args.eval_limit)
        print_metrics("eval_only", metrics)
        out = args.output_dir / "eval_only.json"
        out.write_text(json.dumps({"init_draft": str(args.init_draft), "model": str(args.model),
                                   "draft_vocab": str(args.draft_vocab) if args.draft_vocab else None,
                                   **metrics}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)
        return

    loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_single,
                        generator=generator, num_workers=2)
    params = [p for p in pipe.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_updates = args.epochs * updates_per_epoch
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    warmup = max(1, round(total_updates * args.warmup_ratio))
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_updates)

    start_epoch, opt_step, global_step = 0, 0, 0
    best, history = None, []
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing = set(state["trainable"]) ^ set(pipe.trainable)
        if missing:
            raise ValueError(f"resume state trains a different parameter set: {sorted(missing)[:5]}")
        pipe.model.load_state_dict(state["trainable"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        generator.set_state(state["rng"]["loader"])
        start_epoch, opt_step, global_step = state["epoch"] + 1, state["opt_step"], state["global_step"]
        best, history = state["best"], state["history"]
        print(f"resumed {args.resume}: next epoch {start_epoch}, opt_step {opt_step}", flush=True)

    tracker = Tracker(args, {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    select = f"val/{args.select_metric}"
    if best is None and val_loader is not None:
        # Starting point: the init draft on this target. best/ is only written by
        # an epoch that beats it, so a missing best/ means "keep the init draft".
        initial = evaluate(pipe, val_loader, args.seed, args.eval_limit)
        print_metrics("epoch=-1 (init draft)", initial)
        best = {select: initial[select], "val/loss": initial["val/loss"], "epoch": -1}
        history.append({"epoch": -1, **initial})
        tracker.log(initial, step=0, epoch=-1)

    def save_info(epoch, metrics):
        return {"init_draft": str(args.init_draft), "target_model": str(args.model),
                "draft_vocab": str(args.draft_vocab) if args.draft_vocab else None, "epoch": epoch,
                "opt_step": opt_step, "metrics": metrics,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}

    # Per-slot gradient probe (grad_probe.py) on a fixed set of training rows. It
    # restores every RNG and never touches .grad, so training runs exactly as without it.
    probe_rows: list[int] = []
    probe_dir = args.output_dir / "probe"
    probe_seconds = 0.0
    if args.probe_steps or args.probe_every:
        n_probe = min(args.probe_samples, len(train_set))
        probe_rows = sorted(random.Random(args.seed + 7919).sample(range(len(train_set)), n_probe))
        probe_dir.mkdir(exist_ok=True)
        (probe_dir / "probe_set.json").write_text(json.dumps(
            {"train_subset_rows": probe_rows, "groups": args.probe_groups, "seed": args.seed}) + "\n",
            encoding="utf-8")

    def probe(step: int, epoch: int) -> None:
        nonlocal probe_seconds
        t_start = time.perf_counter()
        py_state = random.getstate()
        summary, tensors = run_probe(
            pipe.model, pipe.call_kwargs,
            lambda i: pipe.draft_batch(collate_single([train_set[probe_rows[i]]])),
            len(probe_rows), groups=args.probe_groups, seed=args.seed, optimizer=optimizer,
            step_check=args.probe_step_check)
        random.setstate(py_state)
        seconds = time.perf_counter() - t_start
        probe_seconds += seconds
        summary.update(opt_step=step, global_step=global_step, epoch=epoch, lr=scheduler.get_last_lr()[0],
                       seconds=seconds)
        torch.save(tensors, probe_dir / f"step_{step:06d}.pt")
        (probe_dir / f"step_{step:06d}.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
        tracker.log(probe_scalars(summary), step=global_step)
        print(f"probe opt_step={step} done in {seconds:.0f}s -> {probe_dir / f'step_{step:06d}.json'}", flush=True)

    pipe.model.train()
    optimizer.zero_grad(set_to_none=True)
    if probe_rows and opt_step == 0 and probe_due(0, args.probe_steps, args.probe_every):
        probe(0, start_epoch)
    stop = False
    started, start_global = time.perf_counter(), global_step
    probe_seconds = 0.0
    for epoch in range(start_epoch, args.epochs):
        window: dict[str, float] = {}
        timing: dict[str, float] = {}  # per-sample stage seconds since the last log line
        epoch_sums: dict[str, float] = {}
        truncated = 0
        for step, raw in enumerate(loader):
            group_start = step - step % args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            loss, metrics, info = pipe.sample(raw)
            t_bwd = cuda_time(args.profile_stages)
            (loss / group_size).backward()
            info["t_draft_bwd"] = cuda_time(args.profile_stages) - t_bwd
            for key in ("t_target", "t_draft_fwd", "t_draft_bwd"):
                timing[key] = timing.get(key, 0.0) + info[key]
            timing["n"] = timing.get("n", 0) + 1
            accumulate_metrics(window, metrics)
            accumulate_metrics(epoch_sums, metrics)
            truncated += int(info["anchors_truncated"])
            global_step += 1
            if step - group_start + 1 == group_size:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % args.log_every == 0 or opt_step == total_updates:
                    logged = {f"train/{k}": v for k, v in reduce_metrics(window).items()}
                    logged.update({"train/lr": scheduler.get_last_lr()[0], "train/grad_norm": grad_norm,
                                   "train/epoch": epoch, "perf/samples_per_sec":
                                   (global_step - start_global)
                                   / max(time.perf_counter() - started - probe_seconds, 1e-9),
                                   "perf/gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1024**3})
                    if args.profile_stages:
                        n_timed = max(timing.get("n", 0), 1)
                        logged.update({f"perf/{k}_ms": 1000 * timing.get(k, 0.0) / n_timed
                                       for k in ("t_target", "t_draft_fwd", "t_draft_bwd")})
                    print(f"epoch={epoch} opt_step={opt_step}/{total_updates} "
                          f"loss={logged.get('train/loss', float('nan')):.4f} "
                          f"eal={logged.get('train/eal', float('nan')):.3f} "
                          f"accept_len={logged.get('train/accept_len', float('nan')):.3f} "
                          f"lr={logged['train/lr']:.2e} grad_norm={grad_norm:.3f} "
                          f"samples/s={logged['perf/samples_per_sec']:.2f} "
                          f"mem={logged['perf/gpu_mem_gb']:.1f}GB", flush=True)
                    if args.profile_stages:
                        print(f"  ms/sample target={logged['perf/t_target_ms']:.0f} "
                              f"draft_fwd={logged['perf/t_draft_fwd_ms']:.0f} "
                              f"draft_bwd={logged['perf/t_draft_bwd_ms']:.0f}", flush=True)
                    tracker.log(logged, step=global_step)
                    window, timing = {}, {}
                if probe_rows and probe_due(opt_step, args.probe_steps, args.probe_every):
                    probe(opt_step, epoch)
                if (args.max_steps and opt_step >= args.max_steps) or (
                        args.stop_after_steps and opt_step >= args.stop_after_steps):
                    stop = True
                    break

        epoch_metrics = {f"train_epoch/{k}": v for k, v in reduce_metrics(epoch_sums).items()}
        epoch_metrics["train_epoch/anchors_truncated_frac"] = truncated / max(step + 1, 1)
        val_metrics = evaluate(pipe, val_loader, args.seed, args.eval_limit) if val_loader else {}
        print_metrics(f"epoch={epoch} train", epoch_metrics)
        if val_metrics:
            print_metrics(f"epoch={epoch} val", val_metrics)
        history.append({"epoch": epoch, **epoch_metrics, **val_metrics})
        tracker.log({**epoch_metrics, **val_metrics}, step=global_step, epoch=epoch)

        if is_improvement(val_metrics, best, select, args.require_loss_decrease):
            best = {select: val_metrics[select], "val/loss": val_metrics["val/loss"], "epoch": epoch}
            # epoch_XX/ keeps every improving epoch; best/ is always the latest of them.
            save_official(pipe.draft, args.output_dir / f"epoch_{epoch:02d}", save_info(epoch, val_metrics))
            save_official(pipe.draft, args.output_dir / "best", save_info(epoch, val_metrics))
            print(f"epoch={epoch} improved: {select}={best[select]:.4f} val/loss={best['val/loss']:.4f}; "
                  f"saved epoch_{epoch:02d}/ and best/", flush=True)
        elif val_metrics:
            print(f"epoch={epoch} not better than epoch {best['epoch']} ({select} {val_metrics[select]:.4f} vs "
                  f"{best[select]:.4f}, val/loss {val_metrics['val/loss']:.4f} vs {best['val/loss']:.4f}); "
                  "best/ unchanged", flush=True)
        last = save_official(pipe.draft, args.output_dir / "last", save_info(epoch, val_metrics))
        save_trainer_state(last / "trainer_state.pt", {
            "trainable": trainable_state(pipe.model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "opt_step": opt_step,
            "global_step": global_step, "best": best, "history": history,
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(), "loader": generator.get_state()},
        })
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if stop:
            break
    if not (args.output_dir / "best").exists() and val_loader is not None:
        print(f"no epoch beat the init draft ({select} up" + (" and val/loss down" if args.require_loss_decrease
              else "") + f"); keep using {args.init_draft}", flush=True)
    tracker.finish()


if __name__ == "__main__":
    main()
