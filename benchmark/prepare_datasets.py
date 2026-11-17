#!/usr/bin/env python3
"""Create collision-free manifests. Annotation sidecars are keyed by relative path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DATASETS = ("omnidocbench", "midv500", "midv-lait")


def collect_images(root):
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def write_manifest(name, root, output_dir, annotations=None):
    root = root.resolve()
    images = collect_images(root)
    if not images:
        raise RuntimeError(f"no supported images found under {root}")
    metadata = json.loads(annotations.read_text(encoding="utf-8")) if annotations else {}
    if not isinstance(metadata, dict):
        raise ValueError("annotation sidecar must be a JSON object keyed by relative image path")
    records = []
    for image in images:
        relative = image.relative_to(root).as_posix()
        extra = metadata.get(relative, {})
        if not isinstance(extra, dict):
            raise ValueError(f"invalid metadata for {relative}")
        record = dict(extra)
        record.update(sample_id=name + "_" + hashlib.sha256(relative.encode()).hexdigest(),
                      dataset=name, relative_path=relative, images=[str(image)])
        # One request is one image/page. A document identity must come from annotations.
        record.setdefault("num_pages", 1)
        record.setdefault("num_images", 1)
        records.append(record)
    manifest = output_dir / f"{name}.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"name": name, "root": str(root), "count": len(records),
            "manifest": str(manifest.resolve()), "annotated_images": sum(bool(metadata.get(r['relative_path'])) for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    for name in DATASETS:
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--{name}-annotations", type=Path,
                            help="JSON sidecar with document_id, document_type, has_table, has_formula, difficulty, ground_truth_fields")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [write_manifest(name, getattr(args, name.replace('-', '_')), args.output_dir,
                              getattr(args, name.replace('-', '_') + '_annotations')) for name in DATASETS]
    (args.output_dir / "dataset_manifest_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(f"{record['name']}: {record['count']} images -> {record['manifest']}")


if __name__ == "__main__":
    main()
