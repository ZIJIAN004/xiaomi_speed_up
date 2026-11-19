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


def _page_metric_eligibility(pages):
    """Return pages with at least one official page-level target category.

    OmniDocBench can contain pages made only of figures, masks, headers,
    footers, page numbers, or ignored annotations. The official evaluator does
    not emit page-level Edit_dist rows for those pages; they are not inference
    failures and must not be used as the denominator of evaluator coverage.
    """
    eligible = set()
    non_evaluable = set()
    for row in pages:
        stem = Path(row['images'][0]).stem
        # Synthetic callers and older manifests lack GT category metadata;
        # preserve the strict historical behavior unless eligibility is known.
        if row.get('evaluator_categories') is None:
            eligible.add(stem)
            continue
        categories = set(row['evaluator_categories'])
        # v1_5 page-level outputs are emitted for the evaluator's canonical
        # targets. Dataset-only annotations such as text_span/text_mask,
        # figures, captions, headers, footers, and ignored equations are not
        # page-level Edit_dist targets in this end-to-end adapter.
        if categories & {'text', 'text_block', 'table', 'equation_isolated', 'display_formula'}:
            eligible.add(stem)
        else:
            non_evaluable.add(stem)
    return eligible, non_evaluable


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
    eligible_stems, non_evaluable_stems = _page_metric_eligibility(pages)
    observed = set(per_page) & expected_stems
    observed_eligible = observed & eligible_stems
    missing_eligible = eligible_stems - observed
    coverage_complete = expected == actual and not missing_eligible
    return dict(status='complete' if not missing and coverage_complete else 'incomplete',
        benchmark_version='1.5', evaluator_version='1.5', gt_version='1.5',
        evaluator=provenance, metrics=scores, missing_metrics=missing,
        required_metrics=list(SCORES)+['overall'], evaluated_samples=len(observed_eligible),
        expected_samples=len(eligible_stems), input_page_count=len(expected_stems),
        evaluator_eligible_pages=sorted(eligible_stems),
        non_evaluable_pages=sorted(non_evaluable_stems),
        sample_coverage_complete=coverage_complete,
        prediction_coverage_complete=actual == {stem + '.md' for stem in expected_stems},
        unobserved_evaluator_pages=sorted(missing_eligible),
        observed_non_evaluable_pages=sorted(observed & non_evaluable_stems),
        evaluation_coverage_source='official page-level Edit_dist union over GT pages with applicable evaluator categories; non-evaluable figure/mask-only pages excluded',
        version_matches_design=True, metric_units={**{k:'ratio_0_1' for k in SCORES}, 'overall':'percent_0_100'},
        per_sample=[dict(sample_id=row['sample_id'],metrics=per_page[Path(row['images'][0]).stem]) for row in pages],
        per_sample_note='official page-level edit scores when emitted; no invented page-level CDM/TEDS',
        overall_definition='100 * ((1 - text_edit) + formula_cdm + table_teds) / 3',
        raw_metric_result=result)
