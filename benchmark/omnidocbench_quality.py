"""Run the explicitly pinned OmniDocBench v1_5 evaluator and normalize its real scores."""
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

from benchmark.metrics import dump_json

SCORES = {
    'text_edit': ('text_block', 'Edit_dist', 'ALL_page_avg'),
    'formula_edit': ('display_formula', 'Edit_dist', 'ALL_page_avg'),
    'formula_cdm': ('display_formula', 'CDM', 'all'),
    'table_teds': ('table', 'TEDS', 'all'),
    'table_edit': ('table', 'Edit_dist', 'ALL_page_avg'),
    'reading_order_edit': ('reading_order', 'Edit_dist', 'ALL_page_avg'),
}


def verify_evaluator(root):
    root = Path(root).resolve()
    if not (root/'pdf_validation.py').is_file():
        raise ValueError('evaluator root must contain pdf_validation.py')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    head = git('rev-parse', 'HEAD')
    refs = []
    for ref in ('refs/heads/v1_5', 'refs/remotes/origin/v1_5'):
        try:
            refs.append(git('rev-parse', '--verify', ref))
        except subprocess.CalledProcessError:
            pass
    if head not in refs:
        raise ValueError('evaluator HEAD must match local v1_5 or origin/v1_5; do not use main/v1.6')
    dirty = git('status', '--porcelain', '--untracked-files=no')
    if dirty:
        raise ValueError('evaluator tracked files differ from pinned v1_5; use a clean evaluator checkout')
    return dict(repository=str(root), branch='v1_5', commit=head, tracked_files_clean=True)


def normalize_scores(result, predictions, pages, work, provenance):
    scores = {}
    for name, keys in SCORES.items():
        value = None
        try:
            value = result[keys[0]]['all']
            for key in keys[1:]:
                value = value[key]
        except (KeyError, TypeError):
            value = None
        scores[name] = float(value) if isinstance(value, (float,int)) and not isinstance(value,bool) and math.isfinite(value) and 0 <= value <= 1 else None
    scores['overall'] = 100*((1-scores['text_edit'])+scores['formula_cdm']+scores['table_teds'])/3 if all(scores[k] is not None for k in ('text_edit','formula_cdm','table_teds')) else None
    missing = [key for key,value in scores.items() if value is None]
    expected = {Path(row['images'][0]).stem+'.md' for row in pages}
    actual = {path.name for path in predictions.glob('*.md')}
    per_page = defaultdict(dict)
    # Read official per-page edit scores without substituting component averages.
    prefix = predictions.name+'_quick_match'
    for name,(element,metric,_) in SCORES.items():
        if metric != 'Edit_dist':
            continue
        path = work/'result'/(prefix+'_'+element+'_per_page_edit.json')
        if path.is_file():
            for image,value in json.loads(path.read_text(encoding='utf-8')).items():
                if isinstance(value,(int,float)) and math.isfinite(value):
                    per_page[Path(image).stem][name] = value
    expected_stems = {Path(row['images'][0]).stem for row in pages}
    observed = set(per_page)&expected_stems
    coverage_complete = expected == actual and observed == expected_stems
    return dict(status='complete' if not missing and coverage_complete else 'incomplete',
        benchmark_version='1.5', evaluator_version='1.5', gt_version='1.5',
        evaluator=provenance, metrics=scores, missing_metrics=missing,
        required_metrics=list(SCORES)+['overall'], evaluated_samples=len(observed),
        expected_samples=len(expected), sample_coverage_complete=coverage_complete,
        prediction_coverage_complete=actual == expected,
        unobserved_evaluator_pages=sorted(expected_stems-observed),
        evaluation_coverage_source='union of actual official per-page Edit_dist outputs, not input file count',
        version_matches_design=True, metric_units={**{k:'ratio_0_1' for k in SCORES}, 'overall':'percent_0_100'},
        per_sample=[dict(sample_id=row['sample_id'],metrics=per_page[Path(row['images'][0]).stem]) for row in pages],
        per_sample_note='official page-level edit scores when emitted; no invented page-level CDM/TEDS',
        overall_definition='100 * ((1 - text_edit) + formula_cdm + table_teds) / 3',
        raw_metric_result=result)
