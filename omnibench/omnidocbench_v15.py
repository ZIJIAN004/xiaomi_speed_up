"""GT-driven OmniDocBench 1.5 whole-page input validation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from benchmark.metrics import dump_json
from benchmark.prepare_datasets import IMAGE_EXTS


def annotation_categories(value):
    """GT labels for diagnostics only, never used as model prompt or OCR text."""
    found = set()
    if isinstance(value,dict) and not value.get('ignore',False):
        if isinstance(value.get('category_type'),str):
            found.add(value['category_type'])
        for child in value.values():
            found.update(annotation_categories(child))
    elif isinstance(value,list):
        for child in value:
            found.update(annotation_categories(child))
    return found


def prepare_pages(dataset, output, expected_pages=1355):
    dataset, output = Path(dataset).resolve(), Path(output)
    if not (dataset/'images').is_dir():
        raise ValueError('dataset root must contain images/')
    gt_path = dataset/'OmniDocBench.json'
    gt = json.loads(gt_path.read_text(encoding='utf-8'))
    if not isinstance(gt, list) or not gt:
        raise ValueError('OmniDocBench.json must be a nonempty page list')
    images = [path for path in (dataset/'images').rglob('*') if path.suffix.lower() in IMAGE_EXTS and path.is_file()]
    by_name = {}
    for image in images:
        if image.name in by_name:
            raise ValueError('ambiguous image basename: '+image.name)
        by_name[image.name] = image
    names = [Path(row['page_info']['image_path'].replace(chr(92), '/')).name for row in gt]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    missing = sorted(set(names)-set(by_name))
    extra = sorted(set(by_name)-set(names))
    report = dict(benchmark_version='1.5', dataset_revision='v1_5', gt_records=len(gt),
                  image_files=len(images), matched_images=len(set(names)&set(by_name)),
                  missing_images=missing, extra_images=extra, duplicate_gt_names=duplicates,
                  matched_fraction=len(set(names)&set(by_name))/len(gt), expected_pages=expected_pages,
                  version_evidence='user-supplied v1_5 dataset; cardinality and GT/image coverage checked, not a release checksum')
    dump_json(output/'data_validation.json', report)
    if duplicates or missing or extra or (expected_pages and len(gt) != expected_pages):
        raise ValueError('OmniDocBench input coverage/count mismatch; inspect data_validation.json')
    if len({Path(name).stem for name in names}) != len(names):
        raise ValueError('page stems must be unique for the official Markdown evaluator')
    pages = []
    for row, name in zip(gt, names):
        sid = 'page_'+hashlib.sha256(name.encode()).hexdigest()
        attributes = row['page_info'].get('page_attribute', {})
        categories = annotation_categories(row['layout_dets']) if isinstance(row.get('layout_dets'),list) else None
        pages.append(dict(sample_id=sid, dataset='omnidocbench', benchmark_version='1.5', request_unit='page',
                          images=[str(by_name[name])], relative_path=name,
                          page_attributes=attributes,
                          evaluator_categories=(sorted(categories) if categories is not None else None),
                          document_type=attributes.get('data_source'),
                          has_table=('table' in categories) if categories is not None else None,
                          has_formula=bool(categories & {'equation_isolated','equation_inline'}) if categories is not None else None))
    write_jsonl(output/'input.jsonl', [dict(images=row['images']) for row in pages])
    write_jsonl(output/'page_manifest.jsonl', pages)
    return pages


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
