#!/usr/bin/env python3
"""Prepare image JSONL manifests for the three vLLM benchmark datasets.

The script does not download datasets. Pass local dataset roots; this avoids
silently benchmarking a partial or wrong copy of a dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_DATASETS = ("omnidocbench", "midv500", "midv-lait")


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def write_manifest(name: str, root: Path, output_dir: Path) -> dict:
    images = collect_images(root)
    if not images:
        raise RuntimeError(f"no supported images found under {root}")
    manifest = output_dir / f"{name}.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for image in images:
            f.write(json.dumps({"images": [str(image.resolve())]}, ensure_ascii=False) + "\n")
    return {"name": name, "root": str(root.resolve()), "count": len(images), "manifest": str(manifest.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create benchmark JSONL manifests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--omnidocbench", type=Path)
    parser.add_argument("--midv500", type=Path)
    parser.add_argument("--midv-lait", dest="midv_lait", type=Path)
    args = parser.parse_args()

    roots = {
        "omnidocbench": args.omnidocbench,
        "midv500": args.midv500,
        "midv-lait": args.midv_lait,
    }
    missing = [name for name in DEFAULT_DATASETS if roots[name] is None]
    if missing:
        parser.error("missing dataset roots: " + ", ".join(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [write_manifest(name, roots[name], output_dir) for name in DEFAULT_DATASETS]
    summary = output_dir / "dataset_manifest_summary.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in records:
        print(f"{record['name']}: {record['count']} images -> {record['manifest']}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
