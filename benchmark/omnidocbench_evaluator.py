"""Invoke a user-prepared local evaluator; never download or install dependencies."""
import json
import os
import re
from pathlib import Path
import subprocess

from benchmark.metrics import dump_json
from benchmark.omnidocbench_quality import normalize_scores
from benchmark.evaluation_runtime import docker_command


def evaluate(root, python, gt, predictions, pages, output, provenance, runtime=None):
    root, predictions, output = Path(root).resolve(), Path(predictions).resolve(), Path(output).resolve()
    work = output/'evaluator_work'
    work.mkdir()
    runtime = runtime or {'backend':'python','python':python}
    docker = runtime['backend'] == 'docker'
    if docker:
        (work/'result').mkdir()
    # Isolate newly generated results while allowing relative evaluator resource reads.
    for asset in root.iterdir():
        if not docker and asset.name not in ('.git', 'result', '__pycache__'):
            (work/asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
    metrics = dict(text_block={'metric':['Edit_dist']}, display_formula={'metric':['Edit_dist','CDM']},
                   table={'metric':['TEDS','Edit_dist']}, reading_order={'metric':['Edit_dist']})
    config = {'end2end_eval':{'metrics':metrics, 'dataset':{
        'dataset_name':'end2end_dataset', 'ground_truth':{'data_path':str(Path(gt).resolve())},
        'prediction':{'data_path':str(predictions)}, 'match_method':'quick_match'}}}
    if docker:
        config['end2end_eval']['dataset']['ground_truth']['data_path'] = '/evaluation/gt.json'
        config['end2end_eval']['dataset']['prediction']['data_path'] = '/evaluation/predictions'
    config_path = output/'evaluator_config.json'
    dump_json(config_path, config)  # JSON is valid YAML for the upstream YAML loader.
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root)+os.pathsep+env.get('PYTHONPATH','')
    command = [python, str(root/'pdf_validation.py'), '--config', str(config_path)]
    if docker:
        command = docker_command(root,work/'result',runtime['image_id'],
            [(gt,'/evaluation/gt.json'),(predictions,'/evaluation/predictions'),
             (config_path,'/evaluation/config.json')])+['/opt/omnidocbench/pdf_validation.py','--config','/evaluation/config.json']
    dump_json(output/'evaluator_invocation.json',dict(command=command,runtime=runtime,evaluator=provenance))
    with (output/'quality_evaluator.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # v1_5 can catch CDM worker exceptions, emit a zero score and still exit 0.
    log_text = (output/'quality_evaluator.log').read_text(encoding='utf-8',errors='replace')
    worker_errors = re.findall(r'Sample \d+ generated an exception:.*',log_text)
    if worker_errors:
        raise RuntimeError('official CDM worker execution errors; not a valid complete evaluation: '+str(worker_errors[:5]))
    result_path = work/'result'/(predictions.name+'_quick_match_metric_result.json')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    quality = normalize_scores(result, predictions, pages, work, provenance)
    quality['evaluation_runtime'] = runtime
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
