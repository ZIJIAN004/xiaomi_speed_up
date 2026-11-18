"""OmniDocBench launch-path tests; no external evaluator, GPU or PaddleX execution."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from benchmark.metrics import dump_json, load_jsonl
from benchmark.omnidocbench_v15 import prepare_pages, prepare_regions, assemble_pages, write_jsonl, REGION_PROMPTS
from benchmark.omnidocbench_quality import normalize_scores, verify_evaluator, SCORES
from benchmark.omnidocbench_evaluator import evaluate, retain_evaluator_scores
from benchmark.processor_audit import prepare_image


def dataset(root, count=2):
    folder = root/'data'
    (folder/'images').mkdir(parents=True)
    gt = []
    for i in range(count):
        name = str(i)+'.png'
        Image.new('RGB',(40,20)).save(folder/'images'/name)
        gt.append({'page_info':{'image_path':'images/'+name,'page_attribute':{}}})
    dump_json(folder/'OmniDocBench.json', gt)
    return folder


def metric_result():
    result = {}
    for name,(element,metric,key) in SCORES.items():
        result.setdefault(element,{'all':{}})['all'][metric] = {key:.1 if metric == 'Edit_dist' else .9}
    return result


def evaluation_evidence(work, pages):
    dump_json(Path(work)/'result/predictions_quick_match_text_block_per_page_edit.json',
              {Path(row['images'][0]).name:.1 for row in pages})


class OmniDocBenchTests(unittest.TestCase):
    def test_launcher_uses_relative_defaults_and_missing_evaluator_fails_before_python(self):
        import os
        import shutil
        bash = shutil.which('bash')
        if not bash:
            self.skipTest('bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'scripts').mkdir()
            shutil.copy2(Path(__file__).resolve().parent.parent/'scripts/run_omnidocbench_v15.sh',root/'scripts/run_omnidocbench_v15.sh')
            (root/'benchmark/data/images').mkdir(parents=True)
            (root/'benchmark/data/OmniDocBench.json').write_text('[]')
            env = dict(os.environ)
            env.pop('OMNIDOC_ROOT',None)
            env.pop('OMNIDOC_EVALUATOR_ROOT',None)
            result = subprocess.run([bash,'scripts/run_omnidocbench_v15.sh'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('Missing v1_5 evaluator:',result.stderr)
            self.assertIn('evaluation_pipeline/OmniDocBench/pdf_validation.py',result.stderr)

    def test_gt_driven_manifest_count_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root)
            output = root/'out'
            pages = prepare_pages(data, output, expected_pages=2)
            self.assertEqual(len(pages),2)
            validation = json.loads((output/'data_validation.json').read_text())
            self.assertEqual(validation['matched_fraction'],1)
            (data/'images/1.png').unlink()
            with self.assertRaisesRegex(ValueError,'coverage'):
                prepare_pages(data, output, expected_pages=2)
            validation = json.loads((output/'data_validation.json').read_text())
            self.assertEqual(validation['matched_fraction'],.5)
            self.assertEqual(validation['missing_images'],['1.png'])

    def test_version_page_count_and_extra_images_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root)
            with self.assertRaises(ValueError):
                prepare_pages(data, root/'out')
            Image.new('RGB',(2,2)).save(data/'images/extra.png')
            with self.assertRaises(ValueError):
                prepare_pages(data, root/'out', expected_pages=2)

    def test_region_crop_before_resize_and_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root,1)
            pages = prepare_pages(data,root/'out',expected_pages=1)
            layout = [dict(index=0,label='text',task_type='text',bbox_2d=[0,0,500,1000]),
                      dict(index=1,label='table',task_type='table',bbox_2d=[500,0,1000,1000]),
                      dict(index=2,label='image',task_type='skip',bbox_2d=[0,0,1000,1000])]
            write_jsonl(root/'layout.jsonl',[dict(images=pages[0]['images'],layout_result=layout)])
            regions,_ = prepare_regions(pages,root/'layout.jsonl',root/'out')
            self.assertEqual(len(regions),2)
            self.assertEqual(regions[1]['prompt'],REGION_PROMPTS['table'])
            self.assertEqual(regions[0]['regions_per_page'],3)
            image, metrics = prepare_image(regions[0]['images'][0],bbox_2d=regions[0]['bbox_2d'])
            self.assertEqual(image.size,(20,20))
            self.assertEqual(metrics['source_page_width'],40)
            self.assertEqual(metrics['original_width'],20)
            self.assertIsNotNone(metrics['region_crop_time_ms'])
            image.close()

    def test_layout_path_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root/'layout.jsonl',[dict(images=['different'],layout_result=[])])
            with self.assertRaisesRegex(ValueError,'exactly'):
                prepare_regions([dict(images=['page'],sample_id='a')],root/'layout.jsonl',root)

    def test_merge_uses_official_page_name_and_preserves_failed_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root,1)
            out = root/'out'
            pages = prepare_pages(data,out,expected_pages=1)
            layout = [dict(index=0,label='text',task_type='text',bbox_2d=[0,0,500,1000]),
                      dict(index=1,label='text',task_type='text',bbox_2d=[500,0,1000,1000])]
            write_jsonl(root/'layout.jsonl',[dict(images=pages[0]['images'],layout_result=layout)])
            regions,layouts = prepare_regions(pages,root/'layout.jsonl',out)
            region_out = root/'regions'
            region_out.mkdir()
            text = region_out/'a.md'
            text.write_text('hello')
            dump_json(region_out/'prediction_mapping.json',[
                dict(sample_id=regions[0]['sample_id'],success=True,prediction_path=str(text)),
                dict(sample_id=regions[1]['sample_id'],success=False)])
            write_jsonl(region_out/'per_sample_metrics.jsonl',[
                dict(sample_id=r['sample_id'],request_start_s=1,request_end_s=2) for r in regions])
            rows,mapping = assemble_pages(pages,regions,layouts,region_out,out)
            self.assertEqual((out/'predictions/0.md').read_text(),'hello')
            self.assertFalse(rows[0]['success'])
            self.assertEqual(rows[0]['failed_regions'],1)
            self.assertTrue(mapping[0]['success'])
            self.assertFalse(mapping[0]['inference_success'])

    def test_real_metric_schema_scales_coverage_and_missing_cdm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root/'predictions'
            predictions.mkdir()
            (predictions/'a.md').write_text('text')
            pages = [dict(sample_id='a',images=['a.png'])]
            result = metric_result()
            evaluation_evidence(root,pages)
            quality = normalize_scores(result,predictions,pages,root,{})
            self.assertEqual(quality['status'],'complete')
            self.assertAlmostEqual(quality['metrics']['overall'],90)
            self.assertEqual(quality['metrics']['table_teds'],.9)
            del result['display_formula']['all']['CDM']
            quality = normalize_scores(result,predictions,pages,root,{})
            self.assertEqual(quality['status'],'incomplete')
            self.assertIsNone(quality['metrics']['overall'])
            self.assertIn('formula_cdm',quality['missing_metrics'])

    def test_prediction_files_alone_are_not_evaluation_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root/'predictions'
            predictions.mkdir()
            (predictions/'a.md').write_text('x')
            quality = normalize_scores(metric_result(),predictions,[dict(sample_id='a',images=['a.png'])],root,{})
            self.assertEqual(quality['status'],'incomplete')
            self.assertEqual(quality['evaluated_samples'],0)
            self.assertTrue(quality['prediction_coverage_complete'])
            self.assertFalse(quality['sample_coverage_complete'])

    def test_evaluator_must_match_v1_5_and_be_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'pdf_validation.py').write_text('# mock')
            with patch('benchmark.omnidocbench_quality.subprocess.check_output',side_effect=['abc','abc','abc','']):
                self.assertEqual(verify_evaluator(root)['commit'],'abc')
            with patch('benchmark.omnidocbench_quality.subprocess.check_output',side_effect=['new','old','old']):
                with self.assertRaisesRegex(ValueError,'v1_5'):
                    verify_evaluator(root)
            with patch('benchmark.omnidocbench_quality.subprocess.check_output',side_effect=['abc','abc','abc',' M metrics.py']):
                with self.assertRaisesRegex(ValueError,'differ'):
                    verify_evaluator(root)

    def test_evaluator_command_is_local_mocked_and_config_calls_real_cdm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root/'eval'
            evaluator.mkdir()
            (evaluator/'pdf_validation.py').write_text('# not executed')
            output = root/'out'
            output.mkdir()
            predictions = output/'predictions'
            predictions.mkdir()
            (predictions/'a.md').write_text('x')
            pages = [dict(sample_id='a',images=['a.png'])]
            def run(command,**kwargs):
                work = Path(kwargs['cwd'])
                dump_json(work/'result/predictions_quick_match_metric_result.json',metric_result())
                evaluation_evidence(work,pages)
                self.assertEqual(command[0],'eval-python')
                self.assertTrue(str(command[1]).endswith('pdf_validation.py'))
            with patch('pathlib.Path.symlink_to'), patch('benchmark.omnidocbench_evaluator.subprocess.run',side_effect=run):
                quality = evaluate(evaluator,'eval-python',root/'gt.json',predictions,pages,output,{})
            config = json.loads((output/'evaluator_config.json').read_text())
            self.assertEqual(config['end2end_eval']['metrics']['display_formula']['metric'],['Edit_dist','CDM'])
            self.assertEqual(quality['status'],'complete')

    def test_evaluator_text_render_cleanup_keeps_all_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root/'evaluator_work/result'
            path = result/'predictions_quick_match_text_block_result.json'
            dump_json(path,[dict(img_id='a',gt='secret gt',pred='long text',metric={'Edit_dist':.1})])
            renders = result/'predictions_quick_match_display_formula/CDM'
            renders.mkdir(parents=True)
            (renders/'render.png').write_bytes(b'generated image')
            keep = result/'predictions_quick_match_metric_result.json'
            dump_json(keep,metric_result())
            before = keep.read_bytes()
            retain_evaluator_scores(root,{'status':'failed'},'sampled')
            self.assertTrue(renders.exists())
            retain_evaluator_scores(root,{'status':'complete'},'sampled')
            self.assertFalse(renders.exists())
            self.assertEqual(keep.read_bytes(),before)
            self.assertEqual(json.loads(path.read_text())[0]['metric'],{'Edit_dist':.1})
            self.assertNotIn('pred',json.loads(path.read_text())[0])


class OmniDocBenchPipelineTests(unittest.TestCase):
    def test_offline_e2e_pipeline_reuses_metrics_and_scores_pages(self):
        self.run_pipeline()

    def test_evaluator_failure_preserves_every_page_and_returns_error(self):
        self.run_pipeline(failure=True)

    def run_pipeline(self, failure=False):
        from benchmark.run_omnidocbench_v15 import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dataset(root,6)
            model = root/'model'
            model.mkdir()
            output = root/'result'
            prompts = []
            def infer(image,args,rid,attempt):
                self.assertEqual(image.size,(40,20))
                prompts.append(args.request_prompt)
                now = time.perf_counter()
                return dict(text='hello', engine_submit_s=now, chat_template_time_ms=.1,
                    effective_max_tokens=100, generated_token_ids=[7], prompt_token_ids=[1,2],
                    finish_reason='stop', server_request_id=rid, events=[dict(time_s=now,token_count=1)],
                    chunk_interval_ms=[], itl_ms=[], itl_source='actual', itl_unavailable_reason=None,
                    first_content_time_s=now, last_content_time_s=now, usage=dict(prompt_tokens=2,completion_tokens=1))
            def rpc(method,args=(),timeout=600):
                return [{'peak_allocated_bytes':100}] if method == 'benchmark_snapshot' else []
            tokenizer = NS(all_special_ids=[],eos_token_id=8,encode=lambda *a,**kw:[7])
            backend = NS(metadata={'backend':'offline'},processor=NS(tokenizer=tokenizer),infer=infer,
                         rpc=rpc,get=lambda endpoint:'' if endpoint == 'metrics' else '{}')
            context = Mock(__enter__=Mock(return_value=backend),__exit__=Mock(return_value=False))
            monitor = NS(errors=[],phase='')
            monitor_context = Mock(__enter__=Mock(return_value=monitor),__exit__=Mock(return_value=False))
            def evaluator(eval_root,python,gt,predictions,pages,destination,provenance):
                self.assertEqual(len(list(predictions.glob('*.md'))),6)
                self.assertEqual({p.name for p in predictions.glob('*.md')},{str(i)+'.md' for i in range(6)})
                if failure:
                    raise RuntimeError('mock evaluator failure')
                evaluation_evidence(destination/'evaluator_work',pages)
                quality = normalize_scores(metric_result(),predictions,pages,destination/'evaluator_work',provenance)
                dump_json(destination/'quality.json',quality)
                return quality
            with patch('benchmark.run_omnidocbench_v15.verify_evaluator',return_value={'branch':'v1_5','commit':'mock'}), patch('benchmark.run_omnidocbench_v15.prepare_pages',side_effect=lambda d,o:prepare_pages(d,o,expected_pages=6)), patch('benchmark.offline_backend.OfflineBackend',return_value=context) as constructor, patch('benchmark.offline_backend._ACTIVE',backend), patch('benchmark.run_benchmark.Monitor',return_value=monitor_context), patch('benchmark.run_benchmark.environment',return_value={}), patch('benchmark.run_omnidocbench_v15.evaluate',side_effect=evaluator), patch('urllib.request.build_opener',side_effect=AssertionError('HTTP forbidden')):
                from contextlib import nullcontext
                with self.assertRaisesRegex(RuntimeError,'quality evaluation') if failure else nullcontext():
                    main(['--dataset',str(data),'--model',str(model),'--output',str(output),
                          '--evaluator-root',str(root/'eval'),'--skip-processor-audit',
                          '--warmup','3','--representatives','3','--retain-samples','0','--retain-worst','0'])
            constructor.assert_called_once()
            child_quality = json.loads((output/'inference/quality.json').read_text())
            if not failure:
                self.assertTrue(set(SCORES).issubset(child_quality['required_metrics']))
                adopted = json.loads((output/'inference/prediction_mapping.json').read_text())
                self.assertTrue(all(Path(row['prediction_path']).is_file() for row in adopted if row.get('prediction_retained')))
                self.assertEqual(len(list((output/'inference/predictions').glob('page_*.md'))),0)
            from benchmark.processor_audit import PROMPT
            self.assertEqual(set(prompts),{PROMPT})
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['quality']['status'],'failed' if failure else 'complete')
            self.assertEqual(summary['page_count'],6)
            self.assertEqual(summary['inference_mode'],'e2e')
            self.assertGreater(summary['steady_page_throughput']['pages_per_second'],0)
            self.assertFalse((output/'layout_results.jsonl').exists())
            self.assertEqual(len(list((output/'inference/predictions').glob('*.md'))),6 if failure else 3)
            records = load_jsonl(output/'inference/per_sample_metrics.jsonl')
            self.assertEqual(len(records),6)
            self.assertTrue(all(row['request_unit'] == 'page' for row in records))
            self.assertEqual(json.loads((output/'environment.txt').read_text())['benchmark_version'],'1.5')
            self.assertEqual(summary['output_retention']['status'],'kept_all' if failure else 'pruned')


if __name__ == '__main__':
    unittest.main()
