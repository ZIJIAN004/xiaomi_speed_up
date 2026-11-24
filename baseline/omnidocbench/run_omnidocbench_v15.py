"""OmniDocBench v1.5 whole-page offline inference and official evaluation; no layout model."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from baseline.harness.metrics import dump_json, load_jsonl, ratio
from baseline.omnidocbench.omnidocbench_v15 import prepare_pages, write_jsonl
from baseline.omnidocbench.omnidocbench_quality import verify_evaluator
from baseline.omnidocbench.omnidocbench_evaluator import evaluate, retain_evaluator_scores
from baseline.harness.output_storage import retain_outputs
from baseline.omnidocbench.evaluation_runtime import preflight, DEFAULT_IMAGE
from baseline.harness.run_benchmark import main as run_pages, parse_args as parse_page_args


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=Path('/home/mr_jiang/models/xiaomi-ocr-0/model'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--evaluator-root', type=Path, default=Path(__file__).resolve().parents[2]/'evaluation_pipeline/OmniDocBench')
    parser.add_argument('--eval-python', default=sys.executable)
    parser.add_argument('--eval-backend',choices=('docker','python'),default='docker')
    parser.add_argument('--eval-image',default=DEFAULT_IMAGE)
    parser.add_argument('--output-retention', choices=('sampled','all'), default='sampled')
    parser.add_argument('--retain-samples', type=int, default=10)
    parser.add_argument('--retain-worst', type=int, default=10)
    parser.add_argument('--retention-seed', type=int, default=42)
    args, shared = parser.parse_known_args(argv)
    if not args.gpu.isdecimal() or not args.model.is_dir():
        parser.error('one numeric GPU and an existing local model directory are required')
    if args.output.exists() or min(args.retain_samples,args.retain_worst) < 0:
        parser.error('output must be new and retention counts nonnegative')
    root = args.output.resolve()
    inference = root/'inference'
    page_args = parse_page_args(['--backend','offline','--manifest',str(root/'page_manifest.jsonl'),
        '--output',str(inference),'--model',str(args.model.resolve()),'--gpu',args.gpu,
        '--workers','8','--output-retention','all','--profiler-dir',str(root/'profiler_raw'),
        '--server-processor-dir',str(root/'processor_raw'), *shared])
    if page_args.workers != 8 or page_args.backend != 'offline' or page_args.quality_command:
        parser.error('requires 8 offline page workers; scoring uses the v1_5 evaluator')
    if (page_args.manifest.resolve() != root/'page_manifest.jsonl' or page_args.output.resolve() != inference
            or Path(page_args.model).resolve() != args.model.resolve()
            or page_args.profiler_dir.resolve() != root/'profiler_raw'
            or page_args.server_processor_dir.resolve() != root/'processor_raw'):
        parser.error('manifest/output/model/evidence paths are managed by this entry')
    page_args.output_retention = 'all'
    provenance = verify_evaluator(args.evaluator_root)
    runtime = preflight(args.evaluator_root,args.eval_backend,args.eval_python,args.eval_image)
    provenance["runtime"] = runtime
    root.mkdir(parents=True)
    pipeline_start = time.perf_counter()
    pages = prepare_pages(args.dataset,root)
    config = dict(benchmark_version='1.5',inference_mode='e2e',evaluator=provenance,
        model=str(args.model.resolve()),gpu_physical=args.gpu,workers=8,backend='offline',
        request_unit='page',score_unit='page',layout_model=None)
    if (args.model/'config.json').is_file():
        from baseline.harness.model_profiles import PROFILES, model_type, profile_for
        if model_type(args.model) in PROFILES:
            config['model_profile'] = profile_for(args.model)['name']
    dump_json(root/'config.json',config)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      VLLM_BENCHMARK_PROCESSOR_DIR=str(root/'processor_raw'))
    for name in ('profiler_raw','processor_raw'):
        (root/name).mkdir()
    page_args.server_pid = os.getpid()
    from baseline.harness.offline_backend import OfflineBackend
    engine_start = time.perf_counter()
    with OfflineBackend(page_args) as backend:
        config['offline_engine'] = backend.metadata
        dump_json(root/'server_startup.json',dict(status='ready',backend='offline',
            cold_start_latency_s=time.perf_counter()-engine_start,
            cold_start_scope='single offline engine construction and initialization'))
        snapshots = backend.rpc('benchmark_snapshot',['false'])
        dump_json(root/'worker_startup.json',snapshots)
        if len(snapshots) != 1 or snapshots[0].get('peak_allocated_bytes') is None:
            raise RuntimeError('expected one observable GPU worker')
        run_pages(page_args)
    populations = {name:(load_jsonl(inference/name) if name.endswith('.jsonl') else
                   json.loads((inference/name).read_text(encoding='utf-8'))) for name in
                   ('per_sample_metrics.jsonl','warmup.json','retry_metrics.json','profile_metrics.json')}
    mapping = json.loads((inference/'prediction_mapping.json').read_text(encoding='utf-8'))
    by_id = {row['sample_id']:row for row in mapping}
    replacements = {}
    for page in pages:
        item = by_id[page['sample_id']]
        target = inference/'predictions'/(Path(page['images'][0]).stem+'.md')
        if item.get('success'):
            source = Path(item['prediction_path'])
            replacements[str(source)] = str(target.resolve())
            source.replace(target)
        else:
            # Keep failures in the evaluation population rather than silently dropping pages.
            target.write_text('',encoding='utf-8')
        item['prediction_path'] = str(target.resolve())
    for name, rows in populations.items():
        for row in rows:
            if row.get('output_path') in replacements:
                row['output_path'] = replacements[row['output_path']]
        if name.endswith('.jsonl'):
            write_jsonl(inference/name,rows)
        else:
            dump_json(inference/name,rows)
    dump_json(inference/'prediction_mapping.json',mapping)
    inference_elapsed = time.perf_counter()-pipeline_start
    eval_start = time.perf_counter()
    try:
        quality = evaluate(args.evaluator_root,args.eval_python,args.dataset/'OmniDocBench.json',
                           inference/'predictions',pages,root,provenance,runtime=runtime)
    except Exception as exc:
        quality = dict(status='failed',benchmark_version='1.5',evaluator_version='1.5',
                       gt_version='1.5',metrics={},reason=str(exc),sample_coverage_complete=False)
    dump_json(root/'quality.json',quality)
    dump_json(inference/'quality.json',quality)
    eval_elapsed = time.perf_counter()-eval_start
    from baseline.harness.collect_results import collect_run
    page_summary = collect_run(inference)
    retention = retain_outputs(inference,populations,mapping,quality,
        mode=args.output_retention,sample_count=args.retain_samples,
        worst_count=args.retain_worst,seed=args.retention_seed)
    evaluator_retention = retain_evaluator_scores(root,quality,args.output_retention)
    page_summary['output_retention'] = retention
    dump_json(inference/'summary.json',page_summary)
    config['environment'] = json.loads((inference/'environment.txt').read_text(encoding='utf-8'))
    dump_json(root/'environment.txt',config['environment'])
    dump_json(root/'config.json',config)
    summary = dict(benchmark_version='1.5',inference_mode='e2e',page_count=len(pages),quality=quality,
        inference_pipeline_elapsed_s=inference_elapsed,evaluation_elapsed_s=eval_elapsed,
        pipeline_timing_scope='validation/cold-start/warmup/steady/retry/profile/prediction naming; excludes evaluation and retention',
        pipeline_pages_per_second=ratio(len(pages),inference_elapsed),
        steady_page_throughput=page_summary['throughput'],output_retention=retention,
        evaluator_output_retention=evaluator_retention,page_report='inference/summary.json',
        acceptance='inference/acceptance_report.json')
    dump_json(root/'summary.json',summary)
    (root/'bottleneck_report.md').write_text(
        '# OmniDocBench v1.5 端到端评测\n\n完整页面直接进入 '+config.get('model_profile',str(args.model))+'，不使用版面检测。\n\n'
        '质量状态：'+quality['status']+'\n\n'
        '请求、Processor、GPU、监控及profiler证据见 [推理报告](inference/bottleneck_report.md)。\n'
        '稳态吞吐排除预热、重试和profile；整条流水线与评分耗时分别记录。\n'
        '质量和实际覆盖见 [quality.json](quality.json)，缺失测量不填0。\n',encoding='utf-8')
    print('OmniDocBench v1.5 e2e completed: '+str(root)+'; quality='+quality['status'],flush=True)
    if quality['status'] != 'complete':
        raise RuntimeError('quality evaluation incomplete/failed; all prediction text preserved; inspect quality.json')


if __name__ == '__main__':
    main()
