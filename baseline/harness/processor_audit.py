"""Separate processor inspection; never mixed into steady-state request timings."""
from __future__ import annotations

import io
import time
from pathlib import Path

from baseline.harness.metrics import PADDING_FIELDS, TOKEN_FIELDS

PROMPT = (
    "Extract all information from the main body of the document image and represent it "
    "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
    "format, formulas in the document should be represented using LATEX format, and the "
    "parsing should be organized according to the reading order."
)


# Longer-side cap applied to every request image (both models). minicpm_image_bound relies on it.
MAX_IMAGE_SIDE = 2048


def prepare_image(path, bbox_2d=None):
    from PIL import Image
    start = time.perf_counter()
    cpu_start = time.thread_time()
    raw = Path(path).read_bytes()
    read_end = time.perf_counter()
    cpu_read_end = time.thread_time()
    with Image.open(io.BytesIO(raw)) as source:
        frames = getattr(source, "n_frames", 1)
        image = source.convert("RGB")
        image.load()
    decode_end = time.perf_counter()
    cpu_decode_end = time.thread_time()
    page_size = image.size
    crop_start = time.perf_counter()
    if bbox_2d is not None:
        from evaluation_pipeline.cropping import crop_region
        cropped = crop_region(image, bbox_2d)
        image.close()
        image = cropped
    crop_ms = (time.perf_counter() - crop_start) * 1000 if bbox_2d is not None else None
    original = image.size
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    prep_end = time.perf_counter()
    return image, {"source_page_width": page_size[0], "source_page_height": page_size[1],
                   "region_crop_time_ms": crop_ms, "region_crop_scope": "nested within preprocess_time_ms",
                   "original_width": original[0], "original_height": original[1],
                   "original_pixels": original[0] * original[1],
                   "image_megapixels": original[0] * original[1] / 1e6,
                   "client_resized_width": image.width, "client_resized_height": image.height,
                   "source_frame_count": frames, "processed_frame_index": 0,
                   "file_io_time_ms": (read_end - start) * 1000,
                   "image_decode_time_ms": (decode_end - read_end) * 1000,
                   "preprocess_time_ms": (prep_end - decode_end) * 1000,
                   "file_io_cpu_time_ms": (cpu_read_end - cpu_start) * 1000,
                   "image_decode_cpu_time_ms": (cpu_decode_end - cpu_read_end) * 1000,
                   "preprocessing_cpu_time_ms": (time.thread_time() - cpu_decode_end) * 1000}


class ProcessorAudit:
    def __init__(self, model, trust_remote_code=False):
        from transformers import AutoConfig, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.tokenizer = self.processor.tokenizer
        self.model = model
        from baseline.harness.model_profiles import image_first
        self.image_first = image_first(model)  # must match the order offline_backend sends

    def describe(self):
        return {"model": self.model, "processor_class": type(self.processor).__name__,
                "tokenizer_class": type(self.tokenizer).__name__,
                "padding_side": self.tokenizer.padding_side,
                "model_max_length": self.tokenizer.model_max_length,
                "image_processor": self.processor.image_processor.to_dict()
                if hasattr(self.processor, "image_processor") else None,
                "chat_template": getattr(self.processor, "chat_template", None),
                "scope": "separate local processor audit; reconcile input length with server usage"}

    def inspect(self, image, prompt=PROMPT):
        start = time.perf_counter()
        from baseline.harness.model_profiles import user_content
        messages = [{"role": "user", "content": user_content(prompt, image, getattr(self, "image_first", False))}]
        template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[template], images=[image], return_tensors="pt")
        elapsed = (time.perf_counter() - start) * 1000
        ids = inputs["input_ids"][0].tolist()
        mask = inputs.get("attention_mask")
        real_ids = ids if mask is None else [i for i, m in zip(ids, mask[0].tolist()) if m]
        result = {k: None for k in (*TOKEN_FIELDS, *PADDING_FIELDS)}
        result.update(processor_time_ms=elapsed, processor_time_source="separate_local_audit_not_server",
                      processor_input_ids_length=len(ids), processor_unpadded_input_length=len(real_ids) if mask is not None else None,
                      processor_attention_padding_tokens=len(ids) - len(real_ids) if mask is not None else None,
                      processor_attention_padding_ratio=(len(ids) - len(real_ids)) / len(ids) if mask is not None and ids else None,
                      prompt_padding_tokens=len(ids) - len(real_ids) if mask is not None else None,
                      prompt_padding_ratio=(len(ids) - len(real_ids)) / len(ids) if mask is not None and ids else None,
                      prompt_length_i=len(real_ids) if mask is not None else None, max_prompt_length=len(ids),
                      prompt_padding_source="actual attention_mask in separate batch=1 processor audit; not scheduler padding",
                      processor_tensor_shapes={k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
                      padding_side=self.tokenizer.padding_side, system_tokens=None,
                      system_tokens_unavailable_reason="chat template may inject implicit system text; no guessed decomposition",
                      user_text_tokens=len(self.tokenizer.encode(prompt, add_special_tokens=False)),
                      processor_serialized_prompt_tokens=len(self.tokenizer.encode(template, add_special_tokens=False)),
                      input_ids_length=len(ids))
        # Count actual expanded image placeholder IDs, never equate raw patches to LLM tokens.
        image_id = getattr(self.config, "image_token_id", None)
        if image_id is None and getattr(self.processor, "image_token", None):
            image_id = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
            if image_id == self.tokenizer.unk_token_id:
                image_id = None
        if image_id is not None and image_id in real_ids:
            visual = real_ids.count(image_id)
            special = set(self.tokenizer.all_special_ids) - {image_id}
            result.update(visual_tokens=visual, visual_tokens_source="expanded_processor_image_token_id_count",
                          special_input_tokens=sum(i in special for i in real_ids),
                          prompt_text_tokens=sum(i != image_id and i not in special for i in real_ids))
        else:
            result["visual_tokens_unavailable_reason"] = "no verified expanded image token ID in processor input"
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            grid = grid.cpu().reshape(-1, 3).tolist()
            result.update(image_grid_thw=grid, image_grid_height=grid[0][1] if len(grid) == 1 else None,
                          image_grid_width=grid[0][2] if len(grid) == 1 else None,
                          image_patch_count=sum(int(t * h * w) for t, h, w in grid),
                          image_patch_count_source="processor.image_grid_thw product")
        # MiniCPM-OCR slicing: one (h, w) patch grid per slice (source thumbnail + up to max_slice_nums tiles).
        sizes = inputs.get("target_sizes")
        if grid is None and sizes is not None:
            sizes = sizes.cpu().reshape(-1, 2).tolist()
            result.update(image_slice_count=len(sizes), image_slice_patch_grids=sizes,
                          image_patch_count=sum(int(h * w) for h, w in sizes),
                          image_patch_count_source="processor.target_sizes product summed over slices")
        pixels = inputs.get("pixel_values")
        result["pixel_values_shape"] = list(pixels.shape) if pixels is not None else None
        # Leave resized pixel geometry/padding null: a patch grid does not prove physical pixel padding.
        result.update(resized_width=None, resized_height=None, resized_pixels=None,
                      num_images=1, processor_padding_scope="offline audit; not scheduler batch padding",
                      batch_padding_unavailable_reason="continuous batching has no observed internal batch membership")
        image_processor = getattr(self.processor, "image_processor", None)
        patch_size = getattr(image_processor, "patch_size", None)
        if grid is not None and len(grid) == 1 and isinstance(patch_size, int):
            height, width = grid[0][1] * patch_size, grid[0][2] * patch_size
            result.update(resized_width=width, resized_height=height, resized_pixels=width * height,
                          resized_geometry_source="processor grid times configured patch size; processed canvas, not padding estimate")
        elif sizes is None and pixels is not None and len(pixels.shape) == 4 and pixels.shape[0] == 1:
            # Not for sliced inputs: MiniCPM packs all slices into one (1, 3, patch, N) strip.
            height, width = int(pixels.shape[-2]), int(pixels.shape[-1])
            result.update(resized_width=width, resized_height=height, resized_pixels=width * height,
                          resized_geometry_source="actual pixel_values spatial dimensions")
        pixel_mask = inputs.get("pixel_attention_mask")
        if pixel_mask is not None and pixels is not None and len(pixels.shape) == 4:
            if tuple(pixel_mask.shape[-2:]) == tuple(pixels.shape[-2:]):
                total = pixel_mask.numel()
                padded = total - int(pixel_mask.sum())
                result.update(pixel_padding_area=padded, pixel_padding_ratio=padded / total if total else 0,
                              pixel_padding_source="actual processor spatial attention mask in separate audit")
        return result
