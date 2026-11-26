"""Glue between MiniCPM-OCR 1.0 and the speculators DSpark draft.

* Builds a speculators ``DSparkDraftModel`` from the official
  ``openbmb/MiniCPM-OCR-1.0-DSpark`` ``config.json`` (a flat Qwen3 config with a
  ``dflash_config`` block, not speculators format) and loads its safetensors.
  Parameter names match one to one, so no renaming happens.
* Attaches the verifier-owned tensors (embedding, reduced verifier head, final
  norm) from the loaded target, so the draft's training targets are exactly the
  target's own logits restricted to the draft vocabulary.
* Extracts the target features that vLLM hands the drafter: aux layer k is the
  residual stream after k decoder layers, i.e. the output of ``layers[k-1]``
  (vLLM ``_maybe_add_hidden_state(aux, layer_idx + 1, hidden, residual)``).
  The pre-norm last hidden state comes from the final norm's input.
* Optionally patches the draft vocabulary in place (``apply_draft_vocab``): same
  size, some official draft ids re-pointed at other target tokens. Only d2t/t2d,
  the draft ``lm_head`` rows and the ``markov_w2`` rows of those ids change, so
  config.json and every tensor shape stay official.
* Saves trained drafts in the official layout (same key set, dtypes and
  ``config.json``), which vLLM's ``qwen3_dspark.py`` loads with
  ``{"method": "dspark"}`` unchanged.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Tensors that the official checkpoint does not contain: vLLM shares the
# embedding with the target, and the verifier head/norm only build training
# targets. t2d is derived from d2t.
VERIFIER_OWNED = frozenset({"embed_tokens.weight", "verifier_lm_head.weight", "verifier_norm.weight"})
DERIVED = frozenset({"t2d"})

# Flat keys of the official config.json that describe the draft decoder layers.
_LAYER_CONFIG_KEYS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "rms_norm_eps", "vocab_size", "max_position_embeddings",
    "attention_bias", "attention_dropout", "hidden_act", "rope_parameters", "layer_types",
    "sliding_window", "use_sliding_window", "initializer_range", "pad_token_id",
    "bos_token_id", "eos_token_id",
)
# dflash_config keys this module understands. Anything else is reported, because
# it may change the architecture in a way the speculators model does not follow.
_KNOWN_DFLASH_KEYS = frozenset({
    "mask_token_id", "target_layer_ids", "projector_type", "markov_head_type", "markov_rank",
    "enable_confidence_head", "confidence_head_with_markov", "confidence_head_alpha",
    "attention_mode", "sample_from_anchor", "block_size",
})


def read_official_config(draft_dir: str | Path) -> dict[str, Any]:
    path = Path(draft_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _dspark_field(official: dict[str, Any], name: str, default: Any) -> Any:
    """Read a DSpark field from dflash_config first, then the top level."""
    dflash = official.get("dflash_config") or {}
    if name in dflash:
        return dflash[name]
    return official.get(name, default)


def build_draft_config(official: dict[str, Any], attn_impl: str = "simple_flex_attention"):
    """speculators ``DSparkSpeculatorConfig`` mirroring the official config.json."""
    from speculators.models.dspark import DSparkSpeculatorConfig
    from transformers import Qwen3Config

    dflash = official.get("dflash_config")
    if not dflash:
        raise ValueError("official config.json has no dflash_config block")
    unknown = sorted(set(dflash) - _KNOWN_DFLASH_KEYS)
    if unknown:
        print(f"[minicpm_dspark] WARNING: unhandled dflash_config keys {unknown}; "
              "check they do not change the draft architecture", flush=True)
    if dflash.get("projector_type", "dspark") != "dspark":
        raise ValueError(f"projector_type {dflash.get('projector_type')!r} is not dspark")
    if dflash.get("attention_mode", "gqa") != "gqa":
        raise ValueError(f"attention_mode {dflash.get('attention_mode')!r} is not supported")
    if official.get("block_size") is None:
        raise ValueError("official config.json has no block_size")

    layer_config = Qwen3Config(**{k: official[k] for k in _LAYER_CONFIG_KEYS if k in official})
    layer_config.tie_word_embeddings = False
    layer_config._attn_implementation = attn_impl  # noqa: SLF001  (as speculators' own builder)
    markov_rank = int(_dspark_field(official, "markov_rank", 256))
    return DSparkSpeculatorConfig(
        transformer_layer_config=layer_config,
        draft_vocab_size=int(official["draft_vocab_size"]),
        block_size=int(official["block_size"]),
        aux_hidden_state_layer_ids=[int(i) for i in dflash["target_layer_ids"]],
        mask_token_id=int(dflash["mask_token_id"]),
        sample_from_anchor=bool(_dspark_field(official, "sample_from_anchor", True)),
        markov_rank=markov_rank,
        markov_head_type=_dspark_field(official, "markov_head_type", "vanilla"),
        enable_confidence_head=bool(_dspark_field(official, "enable_confidence_head", True)),
        confidence_head_with_markov=bool(_dspark_field(official, "confidence_head_with_markov", True)),
    )


def vocab_maps(d2t: torch.Tensor, verifier_vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Delta-encoded d2t (target = i + d2t[i]) -> (target ids per draft id, t2d mask)."""
    target_ids = torch.arange(d2t.numel(), dtype=torch.long) + d2t.long().cpu()
    if target_ids.min() < 0 or target_ids.max() >= verifier_vocab_size:
        raise ValueError("d2t maps outside the verifier vocabulary")
    if target_ids.unique().numel() != target_ids.numel():
        raise ValueError("d2t maps two draft ids to one target id")
    t2d = torch.zeros(verifier_vocab_size, dtype=torch.bool)
    t2d[target_ids] = True
    return target_ids, t2d


def _load_safetensors_dir(draft_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    files = sorted(draft_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {draft_dir}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        part = load_file(str(file))
        overlap = set(part) & set(state)
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)[:5]}")
        state.update(part)
    return state


class OfficialDraft:
    """A DSpark draft loaded from an official-layout directory, plus its save spec."""

    def __init__(self, model, official_config: dict[str, Any], key_dtypes: dict[str, torch.dtype],
                 key_shapes: dict[str, tuple[int, ...]], target_ids: torch.Tensor, source_dir: Path):
        self.model = model
        self.official_config = official_config
        self.key_dtypes = key_dtypes      # exactly the official tensor names -> dtype
        self.key_shapes = key_shapes
        self.target_ids = target_ids      # [draft_vocab] target id of each draft id
        self.source_dir = source_dir
        self.replaced_ids = torch.zeros(0, dtype=torch.long)  # draft ids re-pointed by a vocab patch
        self.random_init_seed: int | None = None  # set when the weights were not loaded (random init)


def load_official_draft(draft_dir: str | Path, attn_impl: str = "simple_flex_attention",
                        random_init_seed: int | None = None) -> OfficialDraft:
    """Build the speculators model and load the official weights (CPU, fp32).

    ``random_init_seed`` keeps the architecture, the vocabulary (d2t) and the save
    layout of ``draft_dir`` but not its weights: every tensor keeps the
    speculators constructor's own initialisation (``post_init``), drawn from this
    seed. The checkpoint key set is still checked against the model.
    """
    from speculators.models.dspark import DSparkDraftModel

    draft_dir = Path(draft_dir)
    official = read_official_config(draft_dir)
    config = build_draft_config(official, attn_impl)
    if random_init_seed is None:
        model = DSparkDraftModel(config)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_init_seed)
            model = DSparkDraftModel(config)
    state = _load_safetensors_dir(draft_dir)
    if "d2t" not in state:
        raise KeyError("official checkpoint has no d2t")
    target_ids, t2d = vocab_maps(state["d2t"], model.verifier_vocab_size)
    if "t2d" in state and not torch.equal(state["t2d"].bool().cpu(), t2d):
        raise ValueError("checkpoint t2d disagrees with its d2t")
    if not bool((target_ids[1:] > target_ids[:-1]).all()):
        print("[minicpm_dspark] note: draft ids are not in ascending target-id order; "
              "verifier head is built by explicit index", flush=True)
    model.load_vocab_mappings(t2d, state["d2t"].long())
    weights = {k: v for k, v in state.items() if k not in DERIVED}
    if random_init_seed is None:
        missing, unexpected = model.load_state_dict(weights, strict=False)
    else:
        own = model.state_dict()
        missing, unexpected = sorted(set(own) - set(weights)), sorted(set(weights) - set(own))
        wrong = sorted(k for k in set(weights) & set(own) if tuple(weights[k].shape) != tuple(own[k].shape))
        if wrong:
            raise ValueError(f"official tensors with another shape than the model: {wrong[:5]}")
    if unexpected:
        raise ValueError(f"official tensors with no model parameter: {sorted(unexpected)}")
    not_verifier = sorted(set(missing) - VERIFIER_OWNED - DERIVED)
    if not_verifier:
        raise ValueError(f"model parameters missing from the official checkpoint: {not_verifier}")
    _check_rotary(model, official)
    draft = OfficialDraft(
        model=model, official_config=official,
        key_dtypes={k: v.dtype for k, v in state.items()},
        key_shapes={k: tuple(v.shape) for k, v in state.items()},
        target_ids=target_ids, source_dir=draft_dir,
    )
    draft.random_init_seed = random_init_seed
    return draft


VOCAB_PATCH_FORMAT = "minicpm_dspark_vocab_patch_v1"


@torch.no_grad()
def apply_draft_vocab(draft: OfficialDraft, patch_path: str | Path) -> dict[str, Any]:
    """Re-point some draft ids at other target tokens (build_kie_draft_vocab.py output).

    The patch records the official d2t it was built from; it must equal the loaded
    one. For each replaced draft id the Markov row ``markov_w2[id]`` is zeroed (the
    old row scored a different token). Its ``lm_head`` row is set from the target
    head in ``attach_verifier``. Tensors indexed by target id (embedding,
    ``markov_w1``) are untouched.
    """
    patch = torch.load(patch_path, map_location="cpu", weights_only=True)
    if patch.get("format") != VOCAB_PATCH_FORMAT:
        raise ValueError(f"{patch_path}: not a {VOCAB_PATCH_FORMAT} file")
    model = draft.model
    current = model.d2t.detach().long().cpu()
    if not torch.equal(patch["base_d2t"].long(), current):
        raise ValueError(f"{patch_path} was built from a different draft vocabulary than {draft.source_dir}")
    d2t = patch["d2t"].long()
    replaced = patch["replaced_draft_ids"].long()
    if d2t.shape != current.shape:
        raise ValueError(f"patched d2t {tuple(d2t.shape)} != draft vocab {tuple(current.shape)}")
    changed = (d2t != current).nonzero().squeeze(-1)
    if not torch.equal(changed, replaced.sort().values):
        raise ValueError("patch replaced_draft_ids do not match the d2t entries it changes")
    target_ids, t2d = vocab_maps(d2t, model.verifier_vocab_size)
    model.load_vocab_mappings(t2d, d2t.to(model.d2t.dtype))
    model.markov_head.markov_w2.weight[replaced] = 0
    draft.target_ids = target_ids
    draft.replaced_ids = replaced
    return {"replaced": int(replaced.numel())}


def _check_rotary(model, official: dict[str, Any]) -> None:
    """The draft uses full-width RoPE (rope_dim = head_dim), unlike the target's partial RoPE."""
    head_dim = int(official.get("head_dim") or official["hidden_size"] // official["num_attention_heads"])
    rope = official.get("rope_parameters") or {}
    expected_dim = int(rope.get("rope_dim", head_dim))
    inv_freq = model.rotary_emb.inv_freq
    if inv_freq.numel() * 2 != expected_dim:
        raise ValueError(f"draft rotary covers {inv_freq.numel() * 2} dims, official rope_dim is {expected_dim}")
    theta = rope.get("rope_theta")
    if theta is not None:
        expected = 1.0 / (float(theta) ** (torch.arange(0, expected_dim, 2, dtype=torch.float64) / expected_dim))
        if not torch.allclose(inv_freq.double().cpu(), expected, rtol=1e-4, atol=0):
            raise ValueError("draft rotary inv_freq does not match official rope_theta")


def find_text_model(target: nn.Module) -> nn.Module:
    """The target's text decoder: the unique module with embed_tokens, layers and norm."""
    text_config = getattr(target.config, "text_config", target.config)
    num_layers = int(text_config.num_hidden_layers)
    found = [
        module for _, module in target.named_modules()
        if isinstance(getattr(module, "layers", None), nn.ModuleList)
        and len(module.layers) == num_layers
        and isinstance(getattr(module, "norm", None), nn.Module)
        and isinstance(getattr(module, "embed_tokens", None), nn.Module)
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one text decoder with {num_layers} layers, found {len(found)}")
    return found[0]


@torch.no_grad()
def attach_verifier(draft: OfficialDraft, target: nn.Module, lm_head_init: str = "checkpoint") -> None:
    """Copy the verifier-owned tensors from the loaded target into the draft.

    ``verifier_norm`` becomes a frozen copy of the target's own final norm, so the
    (1 + w) Qwen3.5 convention is reproduced exactly rather than re-implemented.
    ``lm_head_init="verifier"`` resets the draft head to the target head rows.
    With ``"checkpoint"`` only rows re-pointed by a vocab patch take the target
    head row of their new token (the official row belonged to another token).
    """
    model = draft.model
    text_model = find_text_model(target)
    embed = target.get_input_embeddings().weight
    head = target.get_output_embeddings().weight
    if embed.shape != model.embed_tokens.weight.shape:
        raise ValueError(f"target embedding {tuple(embed.shape)} != draft {tuple(model.embed_tokens.weight.shape)}")
    index = draft.target_ids.to(head.device)
    model.embed_tokens.weight.data = embed.detach().to("cpu", torch.float32).clone()
    model.verifier_lm_head.weight.data = head.detach().index_select(0, index).to("cpu", torch.float32).clone()
    if lm_head_init == "verifier":
        model.lm_head.weight.data = model.verifier_lm_head.weight.data.clone()
    elif lm_head_init == "checkpoint":
        if draft.replaced_ids.numel():
            rows = model.verifier_lm_head.weight.data[draft.replaced_ids]
            model.lm_head.weight.data[draft.replaced_ids] = rows.to(model.lm_head.weight.dtype)
    else:
        raise ValueError(f"unknown lm_head_init {lm_head_init!r}")
    if not bool(torch.isfinite(model.lm_head.weight).all()):
        raise ValueError("draft lm_head is not initialised (random init has no trained head); "
                         "use lm_head_init='verifier'")
    norm = copy.deepcopy(text_model.norm).to("cpu")
    for param in norm.parameters():
        param.requires_grad_(False)
    model.verifier_norm = norm
    model.embed_tokens.weight.requires_grad_(False)
    model.verifier_lm_head.weight.requires_grad_(False)


def configure_trainable(model, train_lm_head: bool) -> list[str]:
    """Freeze verifier-owned tensors; the draft head follows speculators (frozen) unless asked."""
    frozen = {"embed_tokens.weight", "verifier_lm_head.weight"}
    if not train_lm_head:
        frozen.add("lm_head.weight")
    trainable = []
    for name, param in model.named_parameters():
        train = name not in frozen and not name.startswith("verifier_norm.")
        param.requires_grad_(train)
        if train:
            trainable.append(name)
    return trainable


def precast_frozen(model, dtype: torch.dtype, keep: set[str] | frozenset[str] = frozenset()) -> None:
    """Store frozen matrices in the compute dtype (speculators keeps fp32 master copies of trainables).

    ``keep`` names tensors that are saved to the checkpoint (e.g. a frozen lm_head);
    they keep their precision so an untrained save stays bit-identical.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            weight = module.weight
            if f"{name}.weight" in keep:
                continue
            if not weight.requires_grad and weight.is_floating_point() and weight.dtype != dtype:
                weight.data = weight.data.to(dtype)


def save_official(draft: OfficialDraft, out_dir: str | Path, extra_info: dict[str, Any] | None = None) -> Path:
    """Write the draft in the official layout, atomically replacing ``out_dir``.

    Exactly the official key set, shapes and dtypes; non-weight files (config.json
    etc.) are copied from the source directory unchanged.
    """
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    state = draft.model.state_dict()
    tensors = {}
    for name, dtype in draft.key_dtypes.items():
        value = draft.model.t2d if name == "t2d" else state.get(name)
        if value is None:
            raise KeyError(f"model has no tensor for official key {name}")
        value = value.detach().to("cpu", dtype).contiguous()
        if tuple(value.shape) != draft.key_shapes[name]:
            raise ValueError(f"{name}: shape {tuple(value.shape)} != official {draft.key_shapes[name]}")
        tensors[name] = value
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for item in draft.source_dir.iterdir():
        if item.is_file() and not item.name.startswith(".") and not item.name.endswith(".safetensors") \
                and item.name != "model.safetensors.index.json":
            shutil.copy2(item, tmp / item.name)
    save_file(tensors, str(tmp / "model.safetensors"), metadata={"format": "pt"})
    if extra_info is not None:
        (tmp / "dspark_train_info.json").write_text(json.dumps(extra_info, indent=2, default=str) + "\n",
                                                   encoding="utf-8")
    _swap_dir(tmp, out_dir)
    return out_dir


def _swap_dir(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)


class TargetFeatureExtractor:
    """Frozen-target forward that returns exactly the tensors the DSpark drafter consumes.

    Hooks: ``layers[k-1]`` outputs for each aux layer k, the final norm's input
    (pre-norm last hidden state) and the text model's ``position_ids`` (checked to
    be the plain sequential positions vLLM uses for MiniCPM-OCR).
    """

    def __init__(self, target: nn.Module, layer_ids: list[int], use_base_model: bool = True):
        self.target = target
        self.text_model = find_text_model(target)
        num_layers = len(self.text_model.layers)
        if not layer_ids or min(layer_ids) < 1 or max(layer_ids) > num_layers:
            raise ValueError(f"aux layer ids {layer_ids} must lie in [1, {num_layers}]")
        self.layer_ids = list(layer_ids)
        # The base model skips the [T, vocab] logits; the full model is the fallback.
        self.runner = getattr(target, "model", None) if use_base_model else None
        if self.runner is None:
            self.runner = target
        self._captured: dict[str, Any] = {}

    def _hooks(self):
        handles = []

        def layer_hook(k):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                self._captured[f"aux{k}"] = hidden
            return hook

        for k in self.layer_ids:
            handles.append(self.text_model.layers[k - 1].register_forward_hook(layer_hook(k)))

        def norm_pre_hook(_module, args):
            self._captured["last"] = args[0]
        handles.append(self.text_model.norm.register_forward_pre_hook(norm_pre_hook))

        def text_pre_hook(_module, _args, kwargs):
            self._captured["position_ids"] = kwargs.get("position_ids")
        handles.append(self.text_model.register_forward_pre_hook(text_pre_hook, with_kwargs=True))
        return handles

    @torch.no_grad()
    def __call__(self, inputs: dict[str, torch.Tensor], runner: nn.Module | None = None) -> dict[str, torch.Tensor]:
        self._captured = {}
        handles = self._hooks()
        try:
            output = (runner or self.runner)(**inputs, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        missing = [name for name in [f"aux{k}" for k in self.layer_ids] + ["last"] if name not in self._captured]
        if missing:
            raise RuntimeError(f"target hooks did not fire: {missing}")
        seq_len = inputs["input_ids"].shape[1]
        check_sequential_positions(self._captured.get("position_ids"), seq_len)
        aux = torch.cat([self._captured[f"aux{k}"] for k in self.layer_ids], dim=-1)
        if aux.shape[:2] != (1, seq_len):
            raise RuntimeError(f"aux hidden states {tuple(aux.shape)} do not cover input_ids [1, {seq_len}]")
        result = {"hidden_states": aux, "verifier_last_hidden_states": self._captured["last"]}
        logits = getattr(output, "logits", None)
        if logits is not None:
            result["logits"] = logits
        self._captured = {}
        return result


def check_sequential_positions(position_ids: torch.Tensor | None, seq_len: int) -> None:
    """vLLM feeds MiniCPM-OCR plain 0..T-1 positions (minicpmv4_6.py get_mrope_input_positions).

    The draft is trained with the same positions, so a target forward that uses
    anything else would train on features vLLM never produces.
    """
    if position_ids is None:
        return  # the text model builds arange(T) itself
    planes = position_ids.reshape(-1, position_ids.shape[-1])
    expected = torch.arange(seq_len, device=planes.device, dtype=planes.dtype)
    if planes.shape[-1] != seq_len or not bool((planes == expected).all()):
        raise RuntimeError(
            "target text model received non-sequential position_ids; vLLM uses 0..T-1 for "
            f"MiniCPM-OCR, so features would not match serving (shape {tuple(position_ids.shape)})")


def draft_inputs(features: dict[str, torch.Tensor], input_ids: torch.Tensor,
                 loss_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """One unpacked document in the packed-sequence layout DSparkDraftModel.forward expects."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    return {
        "hidden_states": features["hidden_states"],
        "input_ids": input_ids,
        "loss_mask": loss_mask.to(torch.bool),
        "verifier_last_hidden_states": features["verifier_last_hidden_states"],
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long, device=device),
        "position_ids": torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0),
    }


def reduce_metrics(sums: dict[str, float]) -> dict[str, float]:
    """speculators metrics come as ``x_sum`` / ``x_total`` pairs; report x = sum / total."""
    out = {}
    for key, value in sums.items():
        if key.endswith("_sum"):
            name = key[:-4]
            total = sums.get(name + "_total", 0.0)
            if total:
                out[name] = value / total
    return out


def accumulate_metrics(sums: dict[str, float], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        if key.endswith("_sum") or key.endswith("_total"):
            sums[key] = sums.get(key, 0.0) + float(value.detach().float().item()
                                                    if torch.is_tensor(value) else value)
