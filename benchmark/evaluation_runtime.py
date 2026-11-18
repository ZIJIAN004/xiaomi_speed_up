"""Explicit local-Python or Docker runtime for a user-prepared v1_5 checkout."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

DEFAULT_IMAGE = 'sunyuefeng/omnidocbench-env:v1.5'


def docker_command(root, result, image, extra_mounts=()):
    """No implicit pulls, GPU access, network, writable source, or root-owned output."""
    command = ['docker','run','--rm','--pull=never','--network=none']
    if hasattr(os, 'getuid'):
        command += ['--user',str(os.getuid())+':'+str(os.getgid())]
    command += ['--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1',
                '--env','PYTHONPATH=/opt/omnidocbench',
                '--mount',f'type=bind,src={Path(root).resolve()},dst=/opt/omnidocbench,readonly',
                '--mount',f'type=bind,src={Path(result).resolve()},dst=/opt/omnidocbench/result',
                '--workdir','/opt/omnidocbench','--entrypoint','python']
    for source, target in extra_mounts:
        command += ['--mount',f'type=bind,src={Path(source).resolve()},dst={target},readonly']
    return command+[image]


SMOKE = r'''
import json, pathlib, shutil
import dataset, task, metrics
from metrics.cdm_metric import CDM
for tool in ('xelatex','magick','gs','node'):
    if shutil.which(tool) is None:
        raise RuntimeError('missing CDM tool: '+tool)
root = pathlib.Path('result/cdm_preflight')
formula = r'rac{x+1}{y+2}=z'
score = CDM(output_root=str(root)).evaluate(formula,formula,'runtime_smoke')
if score.get('F1_score',0) < .99 or not list(root.rglob('*.png')):
    raise RuntimeError('real identical-formula CDM/render self-test failed: '+str(score))
print('BENCHMARK_EVAL_PREFLIGHT='+json.dumps({'status':'passed','cdm_identical_formula':score}))
'''


def preflight(root, backend='docker', python='python', image=DEFAULT_IMAGE):
    root = Path(root).resolve()
    runtime = dict(backend=backend,python=python)
    if backend not in ('docker','python'):
        raise ValueError('unsupported evaluation backend')
    if backend == 'docker':
        if not image or image.startswith('-') or any(c.isspace() for c in image):
            raise ValueError('invalid evaluator image reference')
        try:
            image_id = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
        except (OSError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Docker image unavailable; prepare '+image+' explicitly before benchmarking') from exc
        runtime.update(image=image,image_id=image_id)
    with tempfile.TemporaryDirectory(prefix='omnidocbench_preflight_') as directory:
        work = Path(directory)
        if backend == 'docker':
            # Execute the immutable local image ID we inspected, not a mutable tag.
            command = docker_command(root,work,runtime['image_id'])+['-c',SMOKE]
            cwd = None
        else:
            for asset in root.iterdir():
                if asset.name not in ('.git','result','__pycache__'):
                    (work/asset.name).symlink_to(asset,target_is_directory=asset.is_dir())
            (work/'result').mkdir()
            command,cwd = [python,'-c',SMOKE],work
        env = dict(os.environ,PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
        try:
            run = subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,check=True,timeout=180)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('evaluator/CDM preflight failed:\n'+(exc.stdout or '')+'\n'+(exc.stderr or '')) from exc
        evidence = [line.split('=',1)[1] for line in run.stdout.splitlines() if line.startswith('BENCHMARK_EVAL_PREFLIGHT=')]
        if len(evidence) != 1:
            raise RuntimeError('evaluator preflight did not produce actual CDM evidence')
        runtime['preflight'] = json.loads(evidence[0])
    return runtime
