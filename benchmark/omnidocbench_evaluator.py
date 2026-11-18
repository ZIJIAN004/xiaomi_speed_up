"""Invoke a user-prepared local evaluator; never download or install dependencies."""
import json
import os
from pathlib import Path
import subprocess

from benchmark.metrics import dump_json
from benchmark.omnidocbench_quality import normalize_scores


def evaluate(root, python, gt, predictions, pages, output, provenance):
    root, predictions, output = Path(root).resolve(), Path(predictions).resolve(), Path(output).resolve()
    work = output/'evaluator_work'
    work.mkdir()
    # Isolate newly generated results while allowing relative evaluator resource reads.
    for asset in root.iterdir():
        if asset.name not in ('.git', 'result', '__pycache__'):
            (work/asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
    metrics = dict(text_block={'metric':['Edit_dist']}, display_formula={'metric':['Edit_dist','CDM']},
                   table={'metric':['TEDS','Edit_dist']}, reading_order={'metric':['Edit_dist']})
    config = {'end2end_eval':{'metrics':metrics, 'dataset':{
        'dataset_name':'end2end_dataset', 'ground_truth':{'data_path':str(Path(gt).resolve())},
        'prediction':{'data_path':str(predictions)}, 'match_method':'quick_match'}}}
    config_path = output/'evaluator_config.json'
    dump_json(config_path, config)  # JSON is valid YAML for the upstream YAML loader.
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root)+os.pathsep+env.get('PYTHONPATH','')
    command = [python, str(root/'pdf_validation.py'), '--config', str(config_path)]
    with (output/'quality_evaluator.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result_path = work/'result'/(predictions.name+'_quick_match_metric_result.json')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    quality = normalize_scores(result, predictions, pages, work, provenance)
    dump_json(output/'quality.json', quality)
    return quality


def retain_evaluator_scores(output, quality, mode):
    """Keep all component scores, discard evaluator text copies/renders after success."""
    import shutil
    output = Path(output)
    report = dict(status='kept_all', removed_render_bytes=0, stripped_text_bytes=0)
    if quality.get('status') != 'complete' or mode == 'all':
        dump_json(output/'evaluator_retention.json', report)
        return report
    result = output/'evaluator_work'/'result'
    for element in ('text_block','display_formula','table','reading_order'):
        path = result/('predictions_quick_match_'+element+'_result.json')
        if path.is_file():
            rows = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(rows, list):
                slim = [{key:row[key] for key in ('img_id','image_name','gt_idx','pred_idx','metric','Edit_num','upper_len') if key in row} for row in rows]
                size = path.stat().st_size
                temporary = path.with_suffix('.tmp')
                dump_json(temporary, slim)
                temporary.replace(path)
                report['stripped_text_bytes'] += size-path.stat().st_size
    renders = result/'predictions_quick_match_display_formula'/'CDM'
    if renders.is_dir() and not renders.is_symlink() and renders.resolve().is_relative_to(result.resolve()):
        report['removed_render_bytes'] = sum(p.stat().st_size for p in renders.rglob('*') if p.is_file() and not p.is_symlink())
        shutil.rmtree(renders)
    report.update(status='scores_only', note='all official scalar/component scores retained; evaluator GT/pred text copies and CDM renders removed')
    dump_json(output/'evaluator_retention.json', report)
    return report
