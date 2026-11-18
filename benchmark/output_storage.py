"""Single-copy predictions and post-evaluation text retention for newly created runs."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import threading

from benchmark.metrics import dump_json

_RAW_LOCK = threading.Lock()


def append_raw(output, record, text):
    path = Path(output) / 'raw_outputs.jsonl'
    row = {key: record[key] for key in ('sample_id', 'phase', 'attempt')}
    row['text'] = text
    encoded = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
    with _RAW_LOCK, path.open('ab') as stream:
        stream.write(encoded)
    record.update(raw_output_path=str(path.resolve()), raw_output_format='shared_jsonl',
                  raw_output_key={key: row[key] for key in ('sample_id', 'phase', 'attempt')})


def finalize_predictions(output, samples, final):
    """Move the adopted file, rather than copying its payload into predictions/."""
    directory = Path(output) / 'predictions'
    directory.mkdir(exist_ok=True)
    mapping = []
    for sample in samples:
        record = final[sample['sample_id']]
        item = dict(sample_id=sample['sample_id'], relative_path=sample.get('relative_path'),
                    image_path=sample['images'][0], success=record['success'],
                    adopted_attempt=record['attempt'])
        if record['success']:
            source = Path(record['output_path'])
            target = directory / (sample['sample_id'] + '.md')
            source.replace(target)
            record['output_path'] = str(target.resolve())
            item['prediction_path'] = record['output_path']
        mapping.append(item)
    return mapping


def _worst_samples(quality, count):
    # Explicit metric direction only; never guess which direction an arbitrary score uses.
    directions = dict(field_accuracy=False, precision=False, recall=False, f1=False,
                      cer=True, wer=True, text_edit=True, formula_cdm=False, table_teds=False)
    observations = defaultdict(list)
    for row in quality.get('per_sample', []):
        if not isinstance(row, dict) or 'sample_id' not in row:
            continue
        metrics = row.get('metrics', row)
        if not isinstance(metrics, dict):
            continue
        for name, lower_better in directions.items():
            value = metrics.get(name)
            if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                observations[name].append((str(row['sample_id']), value if lower_better else -value))
    ranks = {}
    for values in observations.values():
        ordered = sorted(values, key=lambda item: (-item[1], item[0]))
        for rank, (sid, _) in enumerate(ordered):
            severity = 1 - rank / max(1, len(ordered))
            ranks[sid] = max(ranks.get(sid, 0), severity)
    return [sid for sid in sorted(ranks, key=lambda sid: (-ranks[sid], sid))[:count]], sorted(observations)


def _owned_file(path, root):
    path = Path(path)
    # Output roots are new; still never follow an injected symlink out of this run.
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError('refusing to prune a path outside this run or a symlink: ' + str(path))
    return path


def retain_outputs(output, populations, mapping, quality, mode='sampled', sample_count=10,
                   worst_count=10, seed=42, quality_population_size=None, extra_keep_ids=()):
    """Called only after evaluation validation and all metric/report construction."""
    root = Path(output).resolve()
    report = dict(mode=mode, sample_count=sample_count, worst_count=worst_count, seed=seed,
                  status='kept_all', deleted_markdown_files=0, deleted_bytes=0)
    complete = (quality.get('status') == 'complete' and quality.get('sample_coverage_complete') is True
                and quality.get('evaluated_samples') == (len(mapping) if quality_population_size is None else quality_population_size) and not quality.get('missing_metrics')
                and quality.get('version_matches_design') is not False)
    if mode == 'all' or not complete:
        report['reason'] = ('explicit keep-all policy' if mode == 'all' else
                            'quality evaluation unavailable, failed, incomplete, or wrong benchmark version')
        dump_json(root / 'output_retention.json', report)
        return report
    reasons = defaultdict(set)
    ids = sorted(item['sample_id'] for item in mapping)
    for sid in random.Random(seed).sample(ids, min(sample_count, len(ids))):
        reasons[sid].add('seeded_random_sample')
    worst, metrics = _worst_samples(quality, worst_count)
    for sid in worst:
        if sid in ids:
            reasons[sid].add('worst_quality_sample')
    all_records = [r for records in populations.values() for r in records]
    for record in all_records:
        if record.get('phase') == 'profile':
            reasons[record['sample_id']].add('profiler_representative')
        if (not record.get('success') or record.get('length_stopped') or
                record.get('finish_reason') == 'length' or
                record.get('repetition_analysis', {}).get('suspicious') or
                record.get('postprocessed_empty_output')):
            reasons[record['sample_id']].add('failed_truncated_or_repetitive')
    selected_path = root / 'representative_selection.json'
    if selected_path.is_file():
        for sample in json.loads(selected_path.read_text(encoding='utf-8')).get('samples', []):
            reasons[sample['sample_id']].add('selected_representative')
    for sid in extra_keep_ids:
        reasons[sid].add('parent_page_retained')
    keep = set(reasons)
    deletions = {}
    for record in all_records:
        if record.get('output_path') and record['sample_id'] not in keep:
            path = _owned_file(record['output_path'], root)
            if path.is_file():
                deletions[path] = path.stat().st_size
    raw_path = root / 'raw_outputs.jsonl'
    raw_temp = root / 'raw_outputs.jsonl.tmp'
    kept_raw = total_raw = 0
    if raw_path.exists():
        _owned_file(raw_path, root)
        _owned_file(raw_temp, root)
        with raw_path.open('r', encoding='utf-8') as source, raw_temp.open('w', encoding='utf-8') as target:
            for line in source:
                row = json.loads(line)
                total_raw += 1
                if row['sample_id'] in keep:
                    target.write(line)
                    kept_raw += 1
        report['deleted_bytes'] += raw_path.stat().st_size - raw_temp.stat().st_size
        raw_temp.replace(raw_path)
    for path, size in deletions.items():
        path.unlink()
        report['deleted_bytes'] += size
    for record in all_records:
        retained = record['sample_id'] in keep
        for key in ('output_path', 'raw_output_path'):
            if record.get(key) and not retained:
                record[key + '_before_retention'] = record.pop(key)
                record[key] = None
        record['text_retention'] = 'retained' if retained else 'pruned_after_complete_evaluation'
    for item in mapping:
        retained = item['sample_id'] in keep
        if item.get('prediction_path') and not retained:
            item['prediction_path_before_retention'] = item['prediction_path']
            item['prediction_path'] = None
        item['prediction_retained'] = bool(item.get('prediction_path'))
    for filename, records in populations.items():
        temporary = root / (filename + '.tmp')
        if filename.endswith('.jsonl'):
            with temporary.open('w', encoding='utf-8') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        else:
            dump_json(temporary, records)
        temporary.replace(root / filename)
    dump_json(root / 'prediction_mapping.json', mapping)
    report.update(status='pruned', deleted_markdown_files=len(deletions),
                  raw_records_before=total_raw, raw_records_retained=kept_raw,
                  retained_sample_ids=sorted(keep),
                  retention_reasons={sid: sorted(value) for sid, value in sorted(reasons.items())},
                  worst_quality_metrics_used=metrics,
                  worst_quality_note=None if metrics else 'no supported per-sample quality scores; worst-sample selection unavailable',
                  all_scores_preserved=True,
                  reevaluation='full reevaluation requires regenerating pruned predictions')
    dump_json(root / 'output_retention.json', report)
    return report
