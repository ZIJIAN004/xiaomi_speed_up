"""GT-driven OmniDocBench 1.5 input validation and official region assembly."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from benchmark.metrics import dump_json, load_jsonl
from benchmark.prepare_datasets import IMAGE_EXTS

REGION_PROMPTS = {
    'text': 'Extract the text in the image.',
    'table': 'Parse the table in the image into OTSL.',
    'formula': 'Identify the formula in the image and represent it using LATEX format.',
}


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
        pages.append(dict(sample_id=sid, dataset='omnidocbench', benchmark_version='1.5', request_unit='page',
                          images=[str(by_name[name])], relative_path=name,
                          page_attributes=row['page_info'].get('page_attribute', {})))
    write_jsonl(output/'input.jsonl', [dict(images=row['images']) for row in pages])
    write_jsonl(output/'page_manifest.jsonl', pages)
    return pages


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def prepare_regions(pages, layout_path, output):
    layouts = {}
    for row in load_jsonl(layout_path):
        if row.get('error'):
            raise ValueError('layout failure: '+str(row['error']))
        for path in row['images']:
            if path in layouts:
                raise ValueError('duplicate layout page: '+path)
            layouts[path] = row['layout_result']
    if set(layouts) != {row['images'][0] for row in pages}:
        raise ValueError('layout and page input paths must match exactly')
    regions = []
    for page in pages:
        layout = layouts[page['images'][0]]
        for index, region in enumerate(layout):
            if region.get('task_type') == 'skip':
                continue
            task = region.get('task_type', 'text')
            box = region['bbox_2d']
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                raise ValueError('invalid normalized layout bbox: '+str(box))
            regions.append(dict(sample_id=page['sample_id']+'_r'+str(index), dataset='omnidocbench',
                images=page['images'], page_sample_id=page['sample_id'],
                relative_path=page['relative_path'], request_unit='region', region_index=index,
                regions_per_page=len(layout), bbox_2d=box, task_type=task,
                prompt=REGION_PROMPTS.get(task, REGION_PROMPTS['text']),
                has_table=task == 'table', has_formula=task == 'formula'))
    if not regions:
        raise ValueError('no recognizable regions; refusing an empty VLM benchmark')
    write_jsonl(Path(output)/'region_manifest.jsonl', regions)
    dump_json(Path(output)/'layout_summary.json', dict(page_count=len(pages), request_regions=len(regions),
              skipped_regions=sum(r.get('task_type') == 'skip' for rows in layouts.values() for r in rows),
              region_count_per_page={page['sample_id']:len(layouts[page['images'][0]]) for page in pages}))
    return regions, layouts


def assemble_pages(pages, regions, layouts, region_output, output):
    from postprocess.otsl import convert_otsl_to_html
    from postprocess.region import merge_page_to_markdown
    region_output, output = Path(region_output), Path(output)
    by_region = {row['sample_id']:row for row in json.loads((region_output/'prediction_mapping.json').read_text(encoding='utf-8'))}
    region_records = {row['sample_id']:row for row in load_jsonl(region_output/'per_sample_metrics.jsonl')}
    by_page = {page['sample_id']:[] for page in pages}
    for region in regions:
        by_page[region['page_sample_id']].append(region)
    predictions = output/'predictions'
    predictions.mkdir()
    records, mapping = [], []
    for page in pages:
        start = time.perf_counter()
        layout = deepcopy(layouts[page['images'][0]])
        failed = 0
        starts, ends = [], []
        for region in by_page[page['sample_id']]:
            result = by_region[region['sample_id']]
            text = Path(result['prediction_path']).read_text(encoding='utf-8').strip() if result.get('success') else ''
            failed += not result.get('success', False)
            observation = region_records[region['sample_id']]
            starts.append(observation['request_start_s'])
            ends.append(observation['request_end_s'])
            if region['task_type'] == 'table':
                try:
                    text = convert_otsl_to_html(text)
                except Exception:
                    pass  # Match the upstream table-conversion fallback.
            layout[region['region_index']]['merged'] = text
        text = merge_page_to_markdown(layout) or ''
        path = predictions/(Path(page['images'][0]).stem+'.md')
        path.write_text(text, encoding='utf-8')
        records.append(dict(sample_id=page['sample_id'], phase='page_merge', attempt=0,
            success=failed == 0, output_path=str(path.resolve()), failed_regions=failed,
            recognized_regions=len(by_page[page['sample_id']]), layout_regions=len(layout),
            page_merge_write_time_ms=(time.perf_counter()-start)*1000,
            primary_region_span_ms=(max(ends)-min(starts))*1000 if starts else None,
            primary_region_span_scope='first primary region start to last primary region end; not isolated service time; excludes layout and retries'))
        mapping.append(dict(sample_id=page['sample_id'], success=True, inference_success=failed == 0,
                            prediction_path=str(path.resolve()), image_path=page['images'][0]))
    write_jsonl(output/'per_page_metrics.jsonl', records)
    dump_json(output/'prediction_mapping.json', mapping)
    return records, mapping
