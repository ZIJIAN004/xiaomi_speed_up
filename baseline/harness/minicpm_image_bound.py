"""Exact upper bound on MiniCPM-OCR visual tokens per image, for vLLM's encoder-cache sizing.

vLLM sizes the encoder cache as max(max_num_batched_tokens, get_max_image_tokens()), and MiniCPM-OCR's
get_max_image_tokens() evaluates a single 448 x 4032 dummy image (2544 tokens in 4x mode). Real pages can need
more (a 1512 x 2063 page needs 2660), and vLLM then rejects the request.

The token count is a deterministic function of the image's integer (height, width): the processor picks a slice
grid, resizes the overview and the slices, and each piece contributes (h / patch) * (w / patch) / merge tokens
(vLLM's _compute_visual_tokens). The client never sends an image whose longer side exceeds MAX_IMAGE_SIDE
(processor_audit.prepare_image thumbnails to it), so the input domain is finite and an exhaustive evaluation over
every (height, width) in [1, MAX_IMAGE_SIDE]^2 is the exact maximum, not an estimate. It is recomputed from the
checkpoint's own image processor (scale_resolution, patch_size, max_slice_nums, downsample_mode), so a different
config or transformers version yields its own bound; results are cached per configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xiaomi_speed_up" / "minicpm_image_bound.json"


def image_tokens(processor, height, width, downsample_mode, max_slice_nums=None):
    """Visual tokens for one image, computed exactly as vLLM's MiniCPM-V 4.6 _compute_visual_tokens does."""
    s, p = processor.scale_resolution, processor.patch_size
    slices = processor.max_slice_nums if max_slice_nums is None else max_slice_nums
    divisor = p * p * (4 if downsample_mode == "4x" else 16)
    grid = processor.get_sliced_grid((height, width), slices, s)
    if grid is None:
        h, w = processor.find_best_resize((height, width), s, p, downsample_mode=downsample_mode, allow_upscale=True)
        return h * w // divisor
    h, w = processor.find_best_resize((height, width), s, p, downsample_mode=downsample_mode)
    refine_h, refine_w = processor.get_refine_size((height, width), grid, s, p, downsample_mode=downsample_mode,
                                                   allow_upscale=True)
    return h * w // divisor + grid[0] * grid[1] * ((refine_h // grid[0]) * (refine_w // grid[1]) // divisor)


def _key(processor, downsample_mode, max_side):
    import transformers
    return json.dumps({"processor": type(processor).__module__ + "." + type(processor).__qualname__,
                       "transformers": transformers.__version__, "scale_resolution": processor.scale_resolution,
                       "patch_size": processor.patch_size, "max_slice_nums": processor.max_slice_nums,
                       "downsample_mode": downsample_mode, "max_side": max_side}, sort_keys=True)


def max_image_tokens(processor, downsample_mode, max_side, use_cache=True):
    """Exact maximum over every integer image size with both sides in [1, max_side]; returns (tokens, (h, w))."""
    key = _key(processor, downsample_mode, max_side)
    if use_cache:
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8")).get(key)
            if cached:
                return cached["tokens"], tuple(cached["argmax"])
        except (OSError, ValueError):
            pass
    best, argmax = 0, None
    for height in range(1, max_side + 1):
        for width in range(1, max_side + 1):
            tokens = image_tokens(processor, height, width, downsample_mode)
            if tokens > best:
                best, argmax = tokens, (height, width)
    if use_cache:
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
        except (OSError, ValueError):
            data = {}
        data[key] = {"tokens": best, "argmax": list(argmax)}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(temporary, CACHE)
    return best, argmax


def require_patched(model_dir):
    """Fail fast if a MiniCPM-OCR engine was built without the bound (requests would be rejected mid-run)."""
    from baseline.harness.model_profiles import model_type
    try:
        if model_type(model_dir) != "minicpm_ocr":
            return
    except (OSError, ValueError, KeyError):
        return
    import importlib
    info_cls = importlib.import_module("vllm.model_executor.models.minicpm_ocr").MiniCPMOcrProcessingInfo
    if not getattr(info_cls.get_max_image_tokens, "_benchmark_bound", False):
        raise RuntimeError("MiniCPM-OCR max-image-token bound was not installed; vLLM would reject pages above its "
                           "2544-token estimate. Check VLLM_BENCHMARK_MAX_IMAGE_SIDE and server_processor.install().")


def patch_vllm_minicpm_ocr(module, emit=None):
    """Raise MiniCPM-OCR's get_max_image_tokens to the exact bound (never lower it). Opt-in via env var."""
    max_side = os.environ.get("VLLM_BENCHMARK_MAX_IMAGE_SIDE")
    info_cls = getattr(module, "MiniCPMOcrProcessingInfo", None)
    if not max_side or info_cls is None or getattr(info_cls.get_max_image_tokens, "_benchmark_bound", False):
        return
    original = info_cls.get_max_image_tokens
    reported = []

    def get_max_image_tokens(self):
        estimate = original(self)
        mode = self._get_downsample_mode(None) if hasattr(self, "_get_downsample_mode") else "4x"
        bound, argmax = max_image_tokens(self.get_image_processor(), mode, int(max_side))
        if emit is not None and not reported:
            reported.append(True)
            emit(dict(event="max_image_tokens", vllm_estimate=estimate, exact_bound=bound, argmax_hw=list(argmax),
                      max_side=int(max_side), downsample_mode=mode))
        return max(estimate, bound)

    get_max_image_tokens._benchmark_bound = True
    info_cls.get_max_image_tokens = get_max_image_tokens
