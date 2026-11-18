"""Single-copy predictions, shared raw text, and evaluation-gated retention."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.metrics import dump_json, load_jsonl
from benchmark.output_storage import append_raw, finalize_predictions, retain_outputs, _worst_samples
from benchmark.quality import field_quality


def quality(n):
    return dict(status='complete', sample_coverage_complete=True, evaluated_samples=n,
                missing_metrics=[], version_matches_design=None,
                metrics={'field_accuracy': .5}, per_sample=[])


def fixture(root, n=20):
    records, mapping = [], []
    for i in range(n):
        sid = str(i)
        path = root / (sid + '.md')
        path.write_text('prediction '+sid, encoding='utf-8')
        record = dict(sample_id=sid, phase='steady', attempt=0, success=True,
                      finish_reason='stop', output_path=str(path), generated_tokens=10+i)
        append_raw(root, record, 'raw '+sid)
        records.append(record)
        mapping.append(dict(sample_id=sid, success=True, prediction_path=str(path)))
    return {'per_sample_metrics.jsonl': records}, mapping


class OutputStorageTests(unittest.TestCase):
    def test_raw_concurrent_unicode_multiline_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(i):
                row = dict(sample_id=str(i), phase='steady', attempt=0)
                append_raw(root, row, '中文\nline '+str(i))
                return row
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                records = list(pool.map(write, range(200)))
            rows = load_jsonl(root/'raw_outputs.jsonl')
            self.assertEqual(len(rows), 200)
            self.assertEqual(len({r['sample_id'] for r in rows}), 200)
            self.assertEqual(len({r['raw_output_path'] for r in records}), 1)
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertTrue(all(r['text'].startswith('中文\n') for r in rows))

    def test_final_predictions_move_adopted_retry_without_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root/'original.md'
            retry = root/'retry.md'
            original.write_text('original')
            retry.write_text('adopted')
            record = dict(sample_id='a', success=True, attempt=1, output_path=str(retry))
            mapping = finalize_predictions(root, [{'sample_id':'a','images':['input']}], {'a': record})
            self.assertFalse(retry.exists())
            self.assertTrue(original.exists())
            self.assertEqual(Path(mapping[0]['prediction_path']).read_text(), 'adopted')
            self.assertEqual(record['output_path'], mapping[0]['prediction_path'])

    def test_skip_cleanup_for_incomplete_failed_wrong_version_or_keep_all(self):
        for changes, mode in [({'status':'failed'},'sampled'), ({'sample_coverage_complete':False},'sampled'),
                              ({'evaluated_samples':1},'sampled'), ({'missing_metrics':['cer']},'sampled'),
                              ({'version_matches_design':False},'sampled'), ({'status':'unavailable'},'sampled'), ({},'all')]:
            with self.subTest(changes=changes, mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                populations, mapping = fixture(root, 2)
                result = quality(2)
                result.update(changes)
                raw = (root/'raw_outputs.jsonl').read_bytes()
                report = retain_outputs(root, populations, mapping, result, mode, 0, 0)
                self.assertEqual(report['status'], 'kept_all')
                self.assertEqual(len(list(root.glob('*.md'))), 2)
                self.assertEqual((root/'raw_outputs.jsonl').read_bytes(), raw)

    def test_prunes_text_not_metrics_and_keeps_special_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populations, mapping = fixture(root)
            rows = populations['per_sample_metrics.jsonl']
            rows[2]['finish_reason'] = 'length'
            rows[3]['repetition_analysis'] = {'suspicious':True}
            rows[4]['success'] = False
            dump_json(root/'representative_selection.json', {'samples':[{'sample_id':'5'}]})
            result = quality(20)
            result['per_sample'] = [{'sample_id':str(i), 'metrics':{'cer':i}} for i in range(20)]
            dump_json(root/'quality.json', result)
            before = (root/'quality.json').read_bytes()
            report = retain_outputs(root, populations, mapping, result, sample_count=0, worst_count=1)
            self.assertEqual(set(report['retained_sample_ids']), {'2','3','4','5','19'})
            self.assertEqual(report['deleted_markdown_files'], 15)
            self.assertEqual(len(load_jsonl(root/'per_sample_metrics.jsonl')), 20)
            self.assertEqual((root/'quality.json').read_bytes(), before)
            self.assertEqual(len(load_jsonl(root/'raw_outputs.jsonl')), 5)
            self.assertIsNone(mapping[0]['prediction_path'])
            self.assertFalse(mapping[0]['prediction_retained'])
            self.assertIsNone(rows[0]['output_path'])
            self.assertIn('output_path_before_retention', rows[0])
            self.assertEqual(rows[0]['generated_tokens'], 10)

    def test_worst_score_direction_and_missing_scores(self):
        worst, _ = _worst_samples({'per_sample':[
            {'sample_id':'a', 'metrics':{'field_accuracy':1}},
            {'sample_id':'b', 'metrics':{'field_accuracy':0}}]}, 1)
        self.assertEqual(worst, ['b'])
        self.assertEqual(_worst_samples({}, 2), ([], []))

    def test_seeded_random_selection_is_reproducible(self):
        retained = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                populations, mapping = fixture(root)
                report = retain_outputs(root, populations, mapping, quality(20), sample_count=3, worst_count=0)
                retained.append(report['retained_sample_ids'])
                self.assertEqual(len(report['retained_sample_ids']), 3)
        self.assertEqual(retained[0], retained[1])

    def test_outside_path_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent/'run'
            root.mkdir()
            populations, mapping = fixture(root, 1)
            other = parent/'outside.md'
            other.write_text('keep')
            populations['per_sample_metrics.jsonl'][0]['output_path'] = str(other)
            with self.assertRaises(ValueError):
                retain_outputs(root, populations, mapping, quality(1), sample_count=0, worst_count=0)
            self.assertEqual(other.read_text(), 'keep')

    def test_invalid_raw_json_does_not_delete_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populations, mapping = fixture(root, 1)
            (root/'raw_outputs.jsonl').write_text('bad json\n')
            with self.assertRaises(ValueError):
                retain_outputs(root, populations, mapping, quality(1), sample_count=0, worst_count=0)
            self.assertTrue((root/'0.md').is_file())

    def test_profile_analysis_is_preserved_in_population_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populations, mapping = fixture(root, 2)
            profile = dict(populations['per_sample_metrics.jsonl'][0], phase='profile',
                           trace_analysis=[{'kernel_count':12}])
            populations['profile_metrics.json'] = [profile]
            retain_outputs(root, populations, mapping, quality(2), sample_count=0, worst_count=0)
            saved = json.loads((root/'profile_metrics.json').read_text())
            self.assertEqual(saved[0]['trace_analysis'], [{'kernel_count':12}])
            self.assertTrue(Path(saved[0]['output_path']).is_file())

    def test_pruned_mapping_reports_reevaluation_error(self):
        with self.assertRaisesRegex(ValueError, 'pruned'):
            field_quality([{'sample_id':'a','ground_truth_fields':{'id':'1'}}],
                          [{'sample_id':'a','success':True,'prediction_retained':False,'prediction_path':None}])


class OutputRetentionEndToEndTests(unittest.TestCase):
    def test_evaluate_then_prune_and_recollect_without_predictions(self):
        import time
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch
        from PIL import Image
        from benchmark.run_benchmark import main, parse_args
        from benchmark.collect_results import collect_run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'a.png'
            Image.new('RGB', (10,10)).save(image)
            manifest = root/'input.jsonl'
            manifest.write_text('\n'.join(json.dumps(dict(sample_id=str(i), dataset='midv500',
                images=[str(image)], ground_truth_fields={'name':'hello'},
                field_patterns={'name':'(hello)'})) for i in range(20)))
            args = parse_args(['--manifest',str(manifest),'--output',str(root/'run'),
                '--model','model','--warmup','3','--skip-processor-audit',
                '--retain-samples','0','--retain-worst','0','--representatives','3'])
            tokenizer = NS(all_special_ids=[], eos_token_id=8, encode=lambda *a, **kw:[7])
            def infer(image, args, rid, attempt):
                now = time.perf_counter()
                return dict(text='hello', engine_submit_s=now, chat_template_time_ms=.1,
                    effective_max_tokens=100, generated_token_ids=[7], prompt_token_ids=[1,2],
                    finish_reason='stop', server_request_id=rid,
                    events=[dict(time_s=now, token_count=1)], chunk_interval_ms=[], itl_ms=[],
                    itl_source='actual', itl_unavailable_reason=None, first_content_time_s=now,
                    last_content_time_s=now, usage=dict(prompt_tokens=2,completion_tokens=1))
            backend = NS(metadata={'backend':'offline'}, processor=NS(tokenizer=tokenizer),
                         infer=infer, rpc=Mock(return_value=[]),
                         get=lambda endpoint:'' if endpoint == 'metrics' else '{}')
            monitor = NS(errors=[], phase='')
            context = Mock(__enter__=Mock(return_value=monitor), __exit__=Mock(return_value=False))
            with patch('benchmark.offline_backend._ACTIVE', backend), patch('benchmark.run_benchmark.Monitor', return_value=context), patch('urllib.request.build_opener', side_effect=AssertionError('HTTP forbidden')):
                main(args)
            output = root/'run'
            quality_result = json.loads((output/'quality.json').read_text())
            self.assertEqual(quality_result['status'], 'complete')
            self.assertEqual(quality_result['metrics']['cer'], 0)
            self.assertEqual(len(quality_result['per_sample']), 20)
            self.assertEqual(quality_result['per_sample'][0]['metrics']['field_accuracy'], 1)
            report = json.loads((output/'output_retention.json').read_text())
            self.assertEqual(report['status'], 'pruned')
            self.assertEqual(len(report['retained_sample_ids']), 3)
            self.assertGreaterEqual(report['deleted_markdown_files'], 17)
            self.assertEqual(report['deleted_markdown_files'], 23-sum(len(list((output/name).rglob('*.md'))) for name in ('predictions','steady','warmup','retry','profile')))
            self.assertEqual(len(list((output/'predictions').glob('*.md'))), 3)
            self.assertEqual(len(list((output/'steady').rglob('*.md'))), 0)
            self.assertEqual(len(list(output.rglob('*.raw.txt'))), 0)
            self.assertEqual(len(load_jsonl(output/'per_sample_metrics.jsonl')), 20)
            before = (output/'quality.json').read_bytes()
            summary = collect_run(output)
            self.assertEqual(summary['quality']['metrics']['field_accuracy'], 1)
            self.assertEqual(summary['output_retention']['status'], 'pruned')
            self.assertEqual((output/'quality.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
