"""Per-model routing for the baseline harness, keyed by the checkpoint's config.json model_type.

Everything that differs between the benchmarked OCR models lives here; the task itself (pages, prompt,
sampling, concurrency, max_model_len, evaluator) is identical for all of them. Standard library only, so the
launcher can query it with any python before the model environment is chosen:

    python3 -m baseline.harness.model_profiles <model_dir> python   ->  interpreter for that model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ENVS = "/home/mr_jiang/miniforge3/envs"

PROFILES = {
    "qwen3_5": {
        "name": "Xiaomi-OCR-0",
        "python": f"{ENVS}/speed_up/bin/python",
        # vLLM module paths that receive vision/projector CUDA-event ranges.
        "vision_encoder_modules": ["visual", "model.visual"],
        "projector_modules": ["visual.merger", "model.visual.merger"],
        "speculative_methods": [],
        # Official request order (evaluation_pipeline/e2e_img2md.py): prompt text, then image.
        "image_first": False,
        # Image-processor settings copied into config.json for every run.
        "preprocessor_keys": ["patch_size", "merge_size", "temporal_patch_size", "size", "min_pixels", "max_pixels"],
    },
    "minicpm_ocr": {
        "name": "MiniCPM-OCR-1.0",
        "python": f"{ENVS}/minicpm_ocr/bin/python",
        # vLLM never calls vpm.forward: get_vision_hidden_states runs vpm.embeddings / vpm.encoder.layers[i] /
        # post_layernorm piecewise (vit_merger between layers is skipped in 4x mode), then self.merger(...).
        # So the vision stage wraps that model method; merger (final 2x2 downsample MLP) nests inside it.
        "vision_encoder_method": "get_vision_hidden_states",
        "vision_encoder_modules": [],
        "projector_modules": ["merger", "model.merger"],
        "speculative_methods": ["dspark"],
        # Official order (docs/vllm_deploy_en.md "Place the image before the text", SDK dialects.py); the
        # official DSpark draft and our DSpark training data use it too.
        "image_first": True,
        "preprocessor_keys": ["downsample_mode", "slice_mode", "max_slice_nums", "scale_resolution", "patch_size"],
    },
}


def model_type(model_dir) -> str:
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))["model_type"]


def profile_for(model_dir) -> dict:
    kind = model_type(model_dir)
    if kind not in PROFILES:
        raise ValueError(f"no benchmark profile for model_type={kind!r} ({model_dir}); known: {sorted(PROFILES)}")
    return {"model_type": kind, **PROFILES[kind]}


def preprocessor_settings(model_dir, profile) -> dict:
    """Configured preprocessing as shipped with the checkpoint (preprocessor_config.json, then config.json)."""
    found = {}
    for name in ("config.json", "preprocessor_config.json"):
        path = Path(model_dir) / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            found.update({k: data[k] for k in profile["preprocessor_keys"] if k in data})
    return found


def image_first(model_dir) -> bool:
    """Message order for this checkpoint; text-first (the Xiaomi baseline order) when no profile applies."""
    try:
        return bool(profile_for(model_dir).get("image_first", False))
    except (OSError, ValueError, KeyError):
        return False


def user_content(prompt, image, first_image) -> list:
    text, picture = {"type": "text", "text": prompt}, {"type": "image", "image": image}
    return [picture, text] if first_image else [text, picture]


def check_speculative_config(profile, speculative_config) -> None:
    if not speculative_config:
        return
    method = speculative_config.get("method")
    if method not in profile["speculative_methods"]:
        raise ValueError(f"speculative method {method!r} not enabled for {profile['name']}; "
                         f"allowed: {profile['speculative_methods'] or 'none'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m baseline.harness.model_profiles <model_dir> <field>")
    value = profile_for(sys.argv[1])[sys.argv[2]]
    print(value if isinstance(value, str) else json.dumps(value))
