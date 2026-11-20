"""Runtime command tests: no Docker daemon, model, or external evaluator execution."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omnibench.evaluation_runtime import docker_command, preflight, SMOKE
from omnibench.omnidocbench_evaluator import evaluate
from benchmark.metrics import dump_json


class EvaluationRuntimeTests(unittest.TestCase):
    def test_mounts_no_pull_gpu_network_or_shell(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            command = docker_command(root/'source',root/'result','sha256:actual',[(root/'g t.json','/evaluation/gt.json')])
            self.assertIn('--pull=never',command)
            self.assertIn('--network=none',command)
            self.assertNotIn('--gpus',command)
            self.assertEqual(command[-1],'sha256:actual')
            self.assertTrue(any('dst=/opt/omnidocbench,readonly' in x for x in command))
            self.assertTrue(any('g t.json,dst=/evaluation/gt.json,readonly' in x for x in command))
            self.assertNotIn('bash',command)

    def test_preflight_pins_image_and_runs_actual_cdm_program(self):
        with tempfile.TemporaryDirectory() as d:
            completed = SimpleNamespace(stdout='BENCHMARK_EVAL_PREFLIGHT='+json.dumps({'status':'passed','cdm_identical_formula':{'F1_score':1}})+'\n')
            with patch('omnibench.evaluation_runtime.subprocess.check_output',return_value='sha256:immutable\n'), patch('omnibench.evaluation_runtime.subprocess.run',return_value=completed) as run:
                runtime = preflight(Path(d))
            self.assertEqual(runtime['image_id'],'sha256:immutable')
            self.assertIn('sha256:immutable',run.call_args.args[0])
            self.assertIn(SMOKE,run.call_args.args[0])
            self.assertIn(".evaluate(formula,formula",SMOKE)
            self.assertIn("rglob('*.png')",SMOKE)
            self.assertEqual(runtime['preflight']['status'],'passed')

    def test_missing_image_does_not_pull(self):
        with patch('omnibench.evaluation_runtime.subprocess.check_output',side_effect=FileNotFoundError), patch('omnibench.evaluation_runtime.subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError,'prepare .* explicitly'):
                preflight('.')
            run.assert_not_called()

    def test_preflight_rejects_failure_and_missing_evidence(self):
        results = [SimpleNamespace(stdout='nothing'),subprocess.CalledProcessError(1,['docker'],output='render failed',stderr='missing font')]
        for result in results:
            with self.subTest(result=result), patch('omnibench.evaluation_runtime.subprocess.check_output',return_value='sha256:id'), patch('omnibench.evaluation_runtime.subprocess.run') as run:
                if isinstance(result,Exception):
                    run.side_effect = result
                else:
                    run.return_value = result
                with self.assertRaises(RuntimeError):
                    preflight('.')

    def test_docker_maps_config_and_reads_host_scores(self):
        from test_omnidocbench_v15 import metric_result, evaluation_evidence
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source,output,predictions = root/'source',root/'out',root/'predictions'
            source.mkdir(); output.mkdir(); predictions.mkdir()
            (predictions/'a.md').write_text('text')
            pages = [{'sample_id':'a','images':['a.png']}]
            def execute(command,**kwargs):
                self.assertEqual(command[-3:],['/opt/omnidocbench/pdf_validation.py','--config','/evaluation/config.json'])
                self.assertIn('sha256:version',command)
                self.assertIn('--network=none',command)
                dump_json(output/'evaluator_work/result/predictions_quick_match_metric_result.json',metric_result())
                evaluation_evidence(output/'evaluator_work',pages)
            with patch('omnibench.omnidocbench_evaluator.subprocess.run',side_effect=execute):
                quality = evaluate(source,'unused',root/'gt.json',predictions,pages,output,{},runtime={'backend':'docker','image_id':'sha256:version'})
            self.assertEqual(quality['status'],'complete')
            config = json.loads((output/'evaluator_config.json').read_text())
            self.assertEqual(config['end2end_eval']['dataset']['prediction']['data_path'],'/evaluation/predictions')
            self.assertEqual(config['end2end_eval']['dataset']['ground_truth']['data_path'],'/evaluation/gt.json')
            self.assertEqual(config['end2end_eval']['metrics']['display_formula']['metric'],['Edit_dist','CDM'])
            self.assertTrue((output/'evaluator_invocation.json').is_file())

    def test_upstream_worker_error_not_successful_zero_score(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source,output,predictions = root/'source',root/'out',root/'predictions'
            source.mkdir(); output.mkdir(); predictions.mkdir()
            def execute(command,**kwargs):
                kwargs['stdout'].write('Sample 12 generated an exception: broken worker\n')
            with patch('omnibench.omnidocbench_evaluator.subprocess.run',side_effect=execute):
                with self.assertRaisesRegex(RuntimeError,'CDM worker execution errors'):
                    evaluate(source,'unused',root/'gt.json',predictions,[],output,{},runtime={'backend':'docker','image_id':'sha256:id'})

    def test_preflight_failure_before_engine(self):
        from omnibench.run_omnidocbench_v15 import main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root/'model'; model.mkdir()
            with patch('omnibench.run_omnidocbench_v15.verify_evaluator',return_value={}), patch('omnibench.run_omnidocbench_v15.preflight',side_effect=RuntimeError('render check failed')), patch('benchmark.offline_backend.OfflineBackend') as engine:
                with self.assertRaisesRegex(RuntimeError,'render check failed'):
                    main(['--dataset',str(root/'data'),'--model',str(model),'--output',str(root/'out')])
                engine.assert_not_called()
                self.assertFalse((root/'out').exists())

    def test_actual_annotation_metadata_not_sent_as_text(self):
        from test_omnidocbench_v15 import dataset
        from omnibench.omnidocbench_v15 import prepare_pages
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); data = dataset(root,1)
            gt = json.loads((data/'OmniDocBench.json').read_text())
            gt[0]['page_info']['page_attribute'] = {'data_source':'newspaper'}
            gt[0]['layout_dets'] = [{'category_type':'table','text':'never send this'},{'category_type':'text_block','line_with_spans':[{'category_type':'equation_inline'}]}]
            dump_json(data/'OmniDocBench.json',gt)
            page = prepare_pages(data,root/'out',expected_pages=1)[0]
            self.assertEqual(page['document_type'],'newspaper')
            self.assertTrue(page['has_table']); self.assertTrue(page['has_formula'])
            self.assertNotIn('never send this',json.dumps(page))
            self.assertNotIn('prompt',page)


if __name__ == '__main__':
    unittest.main()
