"""Per-slot gradient probe for DSpark block drafts.

Question: DSpark trains every slot of a block in one forward through shared
weights and scales slot k's loss by the fixed decay ``exp(-k / gamma)``. A
weight rescales a slot's gradient but cannot change its direction. The probe
measures, during training, whether later slots' gradients point against
earlier slots' (a conflict that no gamma removes without also starving the
later slots), and separates that from plain gradient noise.

Split of the training loss (exact, per sample)::

    L = sum_k w_k * S_k + alpha * C,      S_k = sum_{t in slot k} m_t * l_t / D

``l_t`` is the per-position compound loss (0.1 CE + 0.9 TV by default), ``m_t``
the loss mask, ``D`` the number of valid positions, ``w_k`` the decay weight,
``C`` the confidence-head BCE exactly as speculators computes it. The probe
takes the gradient of every term ``S_k`` (undecayed) and ``C`` with
``torch.autograd.grad``, so the training ``.grad`` buffers are untouched.

Noise versus signal: the probe samples are dealt into ``groups`` disjoint
groups and each term's gradient is averaged per group. Only Gram matrices of
the group means are kept, per parameter module::

    P[i, j][a, b] = < mean_i g_a , mean_j g_b >

* cross-group products (i != j) are unbiased for <E g_a, E g_b> because the
  two groups' noise is independent: the "signal" view, i.e. the systematic
  directions;
* same-group products (i == j) are what one batch of that size actually sees,
  signal plus noise / batch: the "batch" view (64 samples in 4 groups = the
  16-sample optimizer batch of train_dspark).

Every product is also taken in the AdamW geometry (coordinate i weighted by
1 / (sqrt(v_hat_i) + eps); momentum ignored), the direction AdamW moves along.

Progress ratio of slot k under the joint step G = sum_b c_b g_b (c = decay
weights, then alpha)::

    rho_k = <c_k g_k, G> / |c_k g_k|^2 = 1 + earlier_k + later_k + conf_k

``later_k`` < 0 means the slots after k slow slot k's loss down (first order).
Because the stored gradients are undecayed, any other gamma can be evaluated
from the same Gram matrices without retraining (``what_if``).

Self-checks in every probe: the term gradients must add up to the real
training gradient (reconstruction), and a small real parameter step must move
each term's loss as the Gram matrices predict (first-order validity).
"""
from __future__ import annotations

import contextlib
import itertools
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch
import torch.nn.functional as F

SIDE_HEADS = ("markov_head", "confidence_head")
VIEWS = ("signal", "batch")


# ---------------------------------------------------------------- schedule

def parse_steps(spec: str) -> list[int]:
    steps = sorted({int(s) for s in spec.split(",") if s.strip()})
    if any(s < 0 for s in steps):
        raise ValueError(f"probe steps must be >= 0: {spec!r}")
    return steps


def probe_due(step: int, steps: list[int], every: int) -> bool:
    """``steps`` explicitly, then every ``every`` optimizer steps after the last of them."""
    if step in steps:
        return True
    last = max(steps) if steps else 0
    return every > 0 and step > last and (step - last) % every == 0


# ---------------------------------------------------------------- loss split

def decay_coef(slot_ids: list[int], gamma: float, sample_from_anchor: bool = True) -> torch.Tensor:
    """Fixed-exp-decay weight of each slot (speculators ``dflash_loss_decay``); gamma=inf is uniform."""
    pos = torch.tensor(list(slot_ids), dtype=torch.float64)
    if math.isinf(gamma):
        return torch.where(pos > 0, 1.0, 1.0 if sample_from_anchor else 0.0).to(torch.float64)
    from speculators.losses.utils import dflash_loss_decay
    return dflash_loss_decay(pos, gamma=gamma, sample_from_anchor=sample_from_anchor)


@dataclass
class SlotLosses:
    slot_ids: list[int]
    labels: list[str]
    terms: list[torch.Tensor]   # per-slot losses S_k (no decay), then the confidence term C
    coef: torch.Tensor          # float64 [n_terms]: decay weights, then alpha
    total: torch.Tensor         # sum(coef * terms) == the training loss
    count: torch.Tensor         # [block] valid positions per slot
    correct: torch.Tensor       # [block] greedy draft==target matches per slot


def slot_losses(model, batch: dict[str, torch.Tensor], call_kwargs: dict[str, Any]) -> SlotLosses:
    """``DSparkDraftModel.forward`` + ``compute_metrics`` with the loss split by block slot.

    Mirrors speculators (dspark/core.py forward, dspark/metrics.py compute_metrics);
    the reconstruction check in ``run_probe`` catches any drift from it.
    """
    from speculators.losses import utils as LU
    from speculators.models.dspark.metrics import _masked_decayed_mean

    if call_kwargs.get("per_position_loss_weight", "fixed-exp-decay") != "fixed-exp-decay":
        raise NotImplementedError("the probe splits the fixed-exp-decay loss only")
    if call_kwargs.get("fused_linear"):
        raise NotImplementedError("fused_linear losses are not supported by the probe")
    gamma = float(call_kwargs.get("gamma", 4.0))
    alpha = float(call_kwargs.get("confidence_head_alpha", 1.0))
    loss_config = call_kwargs["loss_config"]
    num_blocks = int(call_kwargs["max_anchors"])
    block = model.block_size
    sfa = bool(model.config.sample_from_anchor)
    input_ids = batch["input_ids"]

    hidden, logits, targets, loss_mask, block_indices = model._backbone_forward(  # noqa: SLF001
        batch["hidden_states"], input_ids, batch["loss_mask"], batch["verifier_last_hidden_states"],
        batch["document_ids"], batch.get("position_ids"), max_anchors=num_blocks)
    block_tokens = input_ids[0, block_indices].view(num_blocks, block)
    prev_ids = block_tokens if sfa else torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    hidden_blocks = hidden.view(num_blocks, block, -1)
    prev_emb = None
    if model.markov_head is not None:
        prev_emb = model.markov_head.prev_embeddings(prev_ids)
        bias = model.markov_head.block_bias(prev_token_ids=prev_ids, hidden_states=hidden_blocks,
                                            prev_emb=prev_emb)
        logits = (logits.view(num_blocks, block, -1) + bias).view(1, num_blocks * block, -1)
    conf_logits = None
    if model.confidence_head is not None:
        if model.config.confidence_head_with_markov and prev_emb is not None:
            feats = torch.cat([hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1)
        else:
            feats = hidden_blocks
        conf_logits = model.confidence_head(feats).reshape(1, num_blocks * block)

    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=logits.device) % block).unsqueeze(0)
    names = [LU._fused_name(fn) for fn, _ in loss_config.values()]  # noqa: SLF001
    if names and all(name is not None for name in names):
        fused = LU._fused_kernel("fused_compound")(logits, targets, names, ("tv",))  # noqa: SLF001
        elementwise = [fused.losses[name] for name in names]
        tv, pred_ids, target_ids = fused.metrics["tv"], fused.pred_ids, fused.target_ids
    else:
        elementwise = [fn(logits, targets) for fn, _ in loss_config.values()]
        with torch.no_grad():
            tv = LU._EAGER_LOSS_FN_MAP["tv"](logits, targets)  # noqa: SLF001
        pred_ids, target_ids = logits.argmax(-1), targets.argmax(-1)

    mask_f = loss_mask.to(torch.float32)
    denom = mask_f.sum(dim=1) + LU._LOSS_REDUCTION_EPS  # noqa: SLF001
    per_pos = sum(float(weight) * term.to(torch.float32)
                  for (_, weight), term in zip(loss_config.values(), elementwise, strict=True)) * mask_f
    slot_ids = [k for k in range(block) if sfa or k > 0]  # DFlash slot 0 is the anchor (weight 0)
    coef = [float(c) for c in decay_coef(slot_ids, gamma, sfa)]
    terms = [((per_pos * (pos_idx == k)).sum(dim=1) / denom).mean() for k in slot_ids]
    labels = [f"slot{k}" for k in slot_ids]
    if conf_logits is not None:
        c_star = (1.0 - tv).detach().to(conf_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(conf_logits, c_star, reduction="none")
        decay = partial(LU.dflash_loss_decay, gamma=gamma, sample_from_anchor=sfa)
        terms.append(_masked_decayed_mean(bce, loss_mask, pos_idx, decay))
        labels.append("conf")
        coef.append(alpha)
    total = sum(c * t for c, t in zip(coef, terms, strict=True))
    with torch.no_grad():
        onehot = F.one_hot(pos_idx[0], block).to(torch.float32)
        hit = ((pred_ids == target_ids) & loss_mask.bool()).to(torch.float32)
        count, correct = (mask_f[0] @ onehot).cpu(), (hit[0] @ onehot).cpu()
    return SlotLosses(slot_ids, labels, terms, torch.tensor(coef, dtype=torch.float64), total, count, correct)


# ---------------------------------------------------------------- parameters

def fine_group(name: str) -> str:
    """Module bucket of a trainable draft parameter."""
    parts = name.split(".")
    if parts[0] in SIDE_HEADS:
        return parts[0]
    if parts[0] == "layers" and len(parts) > 2:
        if parts[2] == "self_attn":
            return f"layers.{parts[1]}.attn"
        if parts[2] == "mlp":
            return f"layers.{parts[1]}.mlp"
        return "norms"
    if parts[0] == "fc":
        return "fc"
    return "norms" if "norm" in name else "other"


@dataclass
class ParamLayout:
    names: list[str]
    params: list[torch.nn.Parameter]
    spans: list[tuple[int, int]]
    numel: int
    groups: dict[str, list[tuple[int, int]]]  # fine group -> merged spans of the flat vector

    def coarse(self) -> dict[str, list[str]]:
        fine = list(self.groups)
        return {"all": fine, "backbone": [g for g in fine if g not in SIDE_HEADS]}


def param_layout(model) -> ParamLayout:
    """Trainable parameters in ``model.parameters()`` order (the optimizer's order)."""
    names, params, spans, groups = [], [], [], {}
    offset = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        span = (offset, offset + param.numel())
        offset = span[1]
        names.append(name)
        params.append(param)
        spans.append(span)
        merged = groups.setdefault(fine_group(name), [])
        if merged and merged[-1][1] == span[0]:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return ParamLayout(names, params, spans, offset, groups)


def adam_scale(optimizer, layout: ParamLayout) -> torch.Tensor | None:
    """Flat 1 / (sqrt(v_hat) + eps) of an AdamW optimizer; None before its first step."""
    if optimizer is None:
        return None
    hyper = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    out = torch.empty(layout.numel, dtype=torch.float32)
    for param, (a, b) in zip(layout.params, layout.spans, strict=True):
        state = optimizer.state.get(param, {})
        if "exp_avg_sq" not in state:
            return None
        group = hyper[id(param)]
        step = float(state["step"])
        v_hat = state["exp_avg_sq"].detach().float() / (1.0 - group["betas"][1] ** step)
        out[a:b] = (1.0 / (v_hat.sqrt() + group["eps"])).reshape(-1).cpu()
    return out


def _flatten(grads, layout: ParamLayout, out: torch.Tensor) -> None:
    for grad, (a, b) in zip(grads, layout.spans, strict=True):
        if grad is None:
            out[a:b].zero_()
        else:
            out[a:b].copy_(grad.reshape(-1))


def _available_ram() -> int | None:
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


# ---------------------------------------------------------------- Gram algebra

def pair_grams(means: torch.Tensor, layout: ParamLayout, scale: torch.Tensor | None = None,
               chunk: int = 1 << 20) -> dict[str, torch.Tensor]:
    """{fine group: [G, G, V, V] float64} with P[i, j, a, b] = <means[i, a], means[j, b]> (scaled)."""
    n_groups, n_terms, _ = means.shape
    rows = n_groups * n_terms
    out = {}
    for name, spans in layout.groups.items():
        acc = torch.zeros(rows, rows, dtype=torch.float64)
        for a, b in spans:
            for s in range(a, b, chunk):
                e = min(b, s + chunk)
                # float64 products: float32 sums over ~1e8 coordinates drift by percents
                x = means[:, :, s:e].reshape(rows, e - s).double()
                if scale is not None:
                    x = x * scale[s:e].double().sqrt()
                acc += x @ x.T
        out[name] = acc.view(n_groups, n_terms, n_groups, n_terms).permute(0, 2, 1, 3).contiguous()
    return out


def coarse_grams(fine: dict[str, torch.Tensor], layout: ParamLayout) -> dict[str, torch.Tensor]:
    return {name: sum(fine[g] for g in members) for name, members in layout.coarse().items()}


def signal_view(pair: torch.Tensor) -> torch.Tensor:
    """Mean cross-group product: unbiased <E g_a, E g_b>."""
    n = pair.shape[0]
    return torch.stack([pair[i, j] for i in range(n) for j in range(n) if i != j]).mean(0)


def batch_view(pair: torch.Tensor) -> torch.Tensor:
    """Mean same-group product: what one batch of the group size sees (signal + noise / size)."""
    return torch.stack([pair[i, i] for i in range(pair.shape[0])]).mean(0)


def split_signals(pair: torch.Tensor) -> list[torch.Tensor]:
    """Cross estimate of every split of the groups into two halves (for the stability spread)."""
    n = pair.shape[0]
    out = []
    for half in itertools.combinations(range(1, n), n // 2 - 1):
        a = (0, *half)
        b = [j for j in range(n) if j not in a]
        cross = [(pair[i, j] + pair[j, i]) / 2 for i in a for j in b]
        out.append(torch.stack(cross).mean(0))
    return out


def full_mean_gram(pair: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Gram of the all-sample means (includes noise; it predicts loss changes on these samples)."""
    w = torch.tensor(sizes, dtype=torch.float64) / sum(sizes)
    return torch.einsum("i,j,ijab->ab", w, w, pair)


def cosine(gram: torch.Tensor) -> torch.Tensor:
    diag = gram.diagonal()
    norm = torch.where(diag > 0, diag, torch.nan).sqrt()
    return gram / (norm[:, None] * norm[None, :])


def decompose(gram: torch.Tensor, coef: torch.Tensor, n_slots: int) -> dict[str, list[float]]:
    """rho_k = 1 + earlier_k + later_k + conf_k for each slot row of ``gram`` (slots in block order)."""
    x, c = gram.double(), coef.double()
    out: dict[str, list[float]] = {"rho": [], "earlier": [], "later": [], "conf": []}
    for a in range(n_slots):
        own = c[a] ** 2 * x[a, a]
        if not bool(own > 0):
            for key in out:
                out[key].append(float("nan"))
            continue
        contrib = c[a] * c * x[a] / own
        earlier, later = float(contrib[:a].sum()), float(contrib[a + 1:n_slots].sum())
        conf = float(contrib[n_slots:].sum())
        out["earlier"].append(earlier)
        out["later"].append(later)
        out["conf"].append(conf)
        out["rho"].append(1.0 + earlier + later + conf)
    return out


def what_if(gram: torch.Tensor, slot_ids: list[int], gammas, alpha: float | None,
            sample_from_anchor: bool = True) -> dict[str, dict[str, list[float]]]:
    """``decompose`` under other decay gammas (alpha None = no confidence term)."""
    out = {}
    for gamma in gammas:
        coef = decay_coef(slot_ids, float(gamma), sample_from_anchor)
        if alpha is not None:
            coef = torch.cat([coef, torch.tensor([alpha], dtype=torch.float64)])
        result = decompose(gram, coef, len(slot_ids))
        result["weight"] = coef[:len(slot_ids)].tolist()
        out[str(gamma)] = result
    return out


def summarize(grams: dict[str, dict[str, torch.Tensor] | None], coef: torch.Tensor,
              n_slots: int) -> dict[str, Any]:
    """{kind: {group: {view: decomposition + cosine, signal also the split spread}}}."""
    out: dict[str, Any] = {}
    for kind, by_group in grams.items():
        if by_group is None:
            out[kind] = None
            continue
        out[kind] = {}
        for group, pair in by_group.items():
            entry = {}
            for view, gram in (("signal", signal_view(pair)), ("batch", batch_view(pair))):
                entry[view] = {**decompose(gram, coef, n_slots), "cos": cosine(gram).tolist()}
            splits = [decompose(g, coef, n_slots)["later"] for g in split_signals(pair)]
            if splits:
                entry["signal"]["later_split_min"] = [min(v) for v in zip(*splits, strict=True)]
                entry["signal"]["later_split_max"] = [max(v) for v in zip(*splits, strict=True)]
            out[kind][group] = entry
    return out


# ---------------------------------------------------------------- probe

def _move(batch: dict[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _term_losses(model, call_kwargs, batches, seed: int, amp, device) -> torch.Tensor:
    total = None
    with torch.no_grad():
        for i, batch in enumerate(batches):
            torch.manual_seed(seed + i)
            with amp:
                values = torch.tensor([float(t) for t in slot_losses(model, _move(batch, device), call_kwargs).terms],
                                      dtype=torch.float64)
            total = values if total is None else total + values
    return total / len(batches)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum()) / den if den > 0 else float("nan")


def run_probe(model, call_kwargs: dict[str, Any], batch_fn: Callable[[int], dict[str, torch.Tensor]],
              n_samples: int, *, groups: int = 4, seed: int = 0, optimizer=None, step_check: bool = True,
              step_frac: float = 0.01, autocast: bool = True,
              log: Callable[[str], None] = print) -> tuple[dict[str, Any], dict[str, Any]]:
    """Probe the current draft on ``n_samples`` samples from ``batch_fn(i)`` (draft_inputs dicts).

    Returns (JSON-able summary, tensors to ``torch.save``). RNG state is restored,
    parameters are restored bit for bit after the step check, ``.grad`` is untouched.
    """
    if groups < 2 or n_samples < 2 * groups:
        raise ValueError("need groups >= 2 and n_samples >= 2 * groups")
    layout = param_layout(model)
    device = layout.params[0].device
    cuda = device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if autocast and cuda else contextlib.nullcontext()
    scale = adam_scale(optimizer, layout)
    group_of = [i % groups for i in range(n_samples)]
    sizes = [group_of.count(g) for g in range(groups)]

    with torch.random.fork_rng(devices=[device] if cuda else []):
        sums = labels = coef = slot_ids = loss_sum = count = correct = recon = None
        flat = torch.empty(layout.numel, dtype=torch.float32, device=device)
        stage = torch.empty(layout.numel, dtype=torch.float32, pin_memory=cuda)
        cached = []
        for i in range(n_samples):
            batch = batch_fn(i)
            torch.manual_seed(seed + i)  # anchor subsampling, identical in every pass over sample i
            with amp:
                split = slot_losses(model, batch, call_kwargs)
            if sums is None:
                labels, coef, slot_ids = split.labels, split.coef, split.slot_ids
                need = groups * len(labels) * layout.numel * 4
                ram = _available_ram()
                if ram is not None and need > 0.8 * ram:
                    raise MemoryError(f"probe needs {need / 2**30:.1f} GiB RAM for {groups} groups x "
                                      f"{len(labels)} terms; {ram / 2**30:.1f} GiB available (lower --probe-groups)")
                sums = torch.zeros(groups, len(labels), layout.numel, dtype=torch.float32)
                loss_sum = torch.zeros(len(labels), dtype=torch.float64)
                count, correct = torch.zeros_like(split.count), torch.zeros_like(split.correct)
            for v, term in enumerate(split.terms):
                grads = torch.autograd.grad(term, layout.params, retain_graph=v < len(split.terms) - 1,
                                            allow_unused=True)
                _flatten(grads, layout, flat)
                stage.copy_(flat)
                sums[group_of[i], v].add_(stage)
                loss_sum[v] += float(term.detach())
            count += split.count
            correct += split.correct
            total_value = float(split.total.detach())
            del split, grads
            if i == 0:
                recon = _reconstruction_check(model, call_kwargs, batch, seed, amp, layout, flat, sums[0], coef,
                                              total_value)
            if step_check:
                cached.append(_move(batch, "cpu"))
            del batch

        for g in range(groups):
            sums[g].div_(sizes[g])
        fine = {"raw": pair_grams(sums, layout)}
        fine["adam"] = pair_grams(sums, layout, scale) if scale is not None else None
        grams = {kind: coarse_grams(by, layout) if by is not None else None for kind, by in fine.items()}

        steps = None
        if step_check:
            steps = _step_check(model, call_kwargs, cached, seed, amp, layout, sums, sizes, coef, grams, scale,
                                step_frac, device, len(slot_ids))
        del sums, cached

    n_slots = len(slot_ids)
    summary = {
        "labels": labels, "slot_ids": slot_ids, "coef": coef.tolist(), "n_samples": n_samples,
        "group_sizes": sizes, "gamma": float(call_kwargs.get("gamma", 4.0)),
        "alpha": float(call_kwargs.get("confidence_head_alpha", 1.0)),
        "sample_from_anchor": bool(model.config.sample_from_anchor),
        "group_numel": {g: sum(b - a for a, b in spans) for g, spans in layout.groups.items()},
        "term_loss": (loss_sum / n_samples).tolist(),
        "slot_acc": [float(c / n) if n > 0 else float("nan") for c, n in zip(correct, count, strict=True)],
        "slot_count": count.tolist(),
        "checks": {"reconstruction": recon, "step": steps},
        "views": summarize(grams, coef, n_slots),
    }
    tensors = {"pair_grams": fine, "meta": {k: summary[k] for k in (
        "labels", "slot_ids", "coef", "group_sizes", "gamma", "alpha", "sample_from_anchor", "group_numel")}}
    log(f"probe: {n_samples} samples, recon cos={recon['cos']:.6f}"
        + "".join(f", step[{k}] corr={v['corr']:.3f} slots act/pred={v['ratio']:.3f}"
                  for k, v in (steps or {}).items() if v))
    return summary, tensors


def _reconstruction_check(model, call_kwargs, batch, seed, amp, layout, flat, vectors, coef, total) -> dict:
    """sum_b c_b g_b of sample 0 against the real training forward's gradient."""
    torch.manual_seed(seed)
    with amp:
        _, loss, _ = model(**batch, **call_kwargs)
    grads = torch.autograd.grad(loss, layout.params, allow_unused=True)
    _flatten(grads, layout, flat)
    true = flat.cpu()
    recon = torch.zeros_like(true)
    for v in range(vectors.shape[0]):
        recon.add_(vectors[v], alpha=float(coef[v]))
    true, recon = true.double(), recon.double()  # float32 norms of ~1e8 entries drift by percents
    dot, nt, nr = float(true @ recon), float(true.norm()), float(recon.norm())
    return {"cos": dot / (nt * nr) if nt * nr > 0 else float("nan"),
            "rel_err": float((recon - true).norm()) / nt if nt > 0 else float("nan"),
            "loss_true": float(loss.detach()), "loss_split": total}


def _step_check(model, call_kwargs, cached, seed, amp, layout, means, sizes, coef, grams, scale, frac, device,
                n_slots):
    """Take a small real step along the joint gradient (raw and AdamW direction) and compare each
    term's loss change with the first-order prediction -eta * (Gram @ coef)."""
    n = sum(sizes)
    direction = torch.zeros(layout.numel, dtype=torch.float32)
    for g, size in enumerate(sizes):
        for v in range(means.shape[1]):
            direction.add_(means[g, v], alpha=float(coef[v]) * size / n)
    base = _term_losses(model, call_kwargs, cached, seed, amp, device)
    total = float(coef @ base)
    backup = [p.detach().clone() for p in layout.params]
    out = {}
    for kind, sc in (("raw", None), ("adam", scale)):
        if grams[kind] is None:
            out[kind] = None
            continue
        gram = full_mean_gram(grams[kind]["all"], sizes)
        quad = float(coef @ gram @ coef)
        if not quad > 0 or not total > 0:
            out[kind] = None
            continue
        eta = frac * total / quad
        step = (direction if sc is None else direction * sc).to(device)
        with torch.no_grad():
            for param, (a, b) in zip(layout.params, layout.spans, strict=True):
                param.add_(step[a:b].view_as(param).to(param.dtype), alpha=-eta)
        after = _term_losses(model, call_kwargs, cached, seed, amp, device)
        with torch.no_grad():
            for param, saved in zip(layout.params, backup, strict=True):
                param.copy_(saved)
        predicted = -eta * (gram @ coef)
        actual = after - base
        # corr / ratio over the slot terms: the confidence target (1 - TV) moves with the
        # parameters, which its detached-target gradient does not predict.
        s = n_slots
        tp, ta = float(coef[:s] @ predicted[:s]), float(coef[:s] @ actual[:s])
        out[kind] = {"eta": eta, "predicted": predicted.tolist(), "actual": actual.tolist(),
                     "slots_predicted": tp, "slots_actual": ta, "ratio": ta / tp if tp else float("nan"),
                     "corr": _pearson(predicted[:s], actual[:s])}
    return out


def probe_scalars(summary: dict[str, Any]) -> dict[str, float]:
    """Tracker scalars: every slot's rho / later / earlier (backbone, signal view, AdamW if available)."""
    kind = "adam" if summary["views"].get("adam") else "raw"
    view = summary["views"][kind]["backbone"]["signal"]
    out = {}
    for k, rho, later, earlier in zip(summary["slot_ids"], view["rho"], view["later"], view["earlier"], strict=True):
        out[f"probe/rho/slot{k:02d}"] = rho
        out[f"probe/later/slot{k:02d}"] = later
        out[f"probe/earlier/slot{k:02d}"] = earlier
    # mean cosine between the first and the last third of the block (one number for "early vs late")
    n = len(summary["slot_ids"])
    third = max(1, n // 3)
    cos = view["cos"]
    pairs = [cos[i][j] for i in range(third) for j in range(n - third, n)]
    pairs = [c for c in pairs if c is not None and math.isfinite(c)]
    out["probe/cos_early_late"] = sum(pairs) / len(pairs) if pairs else None
    out["probe/later_mean_early"] = sum(view["later"][:third]) / third
    out["probe/recon_cos"] = summary["checks"]["reconstruction"]["cos"]
    for name, check in (summary["checks"]["step"] or {}).items():
        if check:
            out[f"probe/step_corr_{name}"] = check["corr"]
            out[f"probe/step_ratio_{name}"] = check["ratio"]
    return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(v)}
