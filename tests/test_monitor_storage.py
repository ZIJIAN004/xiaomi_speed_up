"""Reduced-write monitoring tests; no GPU or vLLM required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.collect_results import prometheus_summary
from benchmark.metrics import stats
from benchmark.monitor import Monitor
from benchmark.monitor_storage import ScalarArchive, ScalarStore


def record(t, phase='steady', end_phase=None, value=1.):
    return dict(monotonic_s=t, sample_end_monotonic_s=t+.1, sampling_duration_s=.1,
                wall_time_s=100+t, phase=phase, phase_end=end_phase or phase,
                gpu_utilization_percent=value, pinned_memory_bytes=None,
                system_memory={'MemAvailable': 1000})


class StorageTests(unittest.TestCase):
    def test_roundtrip_missing_new_series_and_quantiles(self):
        store = ScalarStore()
        for i in range(100):
            store.append(record(i, value=i), {('vllm:running', '{model="a"}'): i})
        store.append(record(101, 'profile'), {('new', ''): 8})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scalars.zip'
            store.save(path)
            archive = ScalarArchive(path)
            rows = list(archive.system_records())
            self.assertEqual(len(rows), 101)
            self.assertIsNone(rows[0]['pinned_memory_bytes'])
            self.assertEqual(rows[-1]['phase'], 'profile')
            actual = dict(archive.prometheus_statistics(0, 100))
            self.assertEqual(actual[('vllm:running', '{model="a"}')], stats(range(100)))
            self.assertEqual(actual[('new', '')]['count'], 0)
            self.assertAlmostEqual(actual[('vllm:running', '{model="a"}')]['p99'], 98.01)

    def test_checkpoint_online_moments_missing_and_crash_scope(self):
        store = ScalarStore()
        for i in range(4):
            store.append(record(i, value=i), {('x', ''): i} if i else {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'monitor_summary.json'
            store.checkpoint(path, 1, [])
            data = json.loads(path.read_text())
            row = next(r for r in data['statistics'] if r['series'] == ['prometheus','x',''])
            self.assertEqual(row['count'], 3)
            self.assertEqual(row['missing'], 1)
            self.assertEqual(row['mean'], 2)
            self.assertFalse(data['complete'])
            self.assertIn('lost', data['crash_scope'])
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_phase_window_and_counter_histogram_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'monitor').mkdir()
            (root/'metrics_before.prom').write_text('lat_sum 10\nlat_count 2\nx_total 9\n')
            (root/'metrics_after.prom').write_text('lat_sum 30\nlat_count 6\nx_total 1\n')
            store = ScalarStore()
            for t, phase, end_phase, value in [(0,'warmup',None,900), (1,'steady','transition',800),
                                              (2,'steady',None,2), (3,'steady',None,4),
                                              (4,'profile',None,700), (8,'steady',None,600)]:
                store.append(record(t,phase,end_phase), {('gauge',''): value})
            store.save(root/'monitor/scalars.zip')
            rows, hist = prometheus_summary(root, [], 1, 4)
            gauge = next(r for r in rows if r['metric'] == 'gauge')
            self.assertEqual(gauge['sample_statistics'], stats([2,4]))
            reset = next(r for r in rows if r['metric'] == 'x_total')
            self.assertTrue(reset['counter_reset'])
            self.assertIsNone(reset['delta'])
            self.assertEqual(hist['lat_sum']['mean'], 5)

    def test_legacy_prometheus_files_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'monitor').mkdir()
            (root/'monitor/old.prom').write_text('gauge 8\n')
            sample = record(1)
            sample['prometheus_file'] = 'old.prom'
            rows, _ = prometheus_summary(root, [sample], 0, 2)
            self.assertEqual(rows[0]['sample_statistics']['mean'], 8)

    def test_samples_do_not_create_per_round_files(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(directory, 'offline://single', '1')
            snapshot = dict(server_process_cpu_percent=2, server_process_rss_bytes=8,
                            system_memory={}, pinned_memory_bytes=None)
            gpu = SimpleNamespace(stdout='1,2,3,4,5,6,7,8\n')
            with patch('benchmark.monitor.process_snapshot', side_effect=lambda *a: (snapshot.copy(),None)), patch('benchmark.monitor.subprocess.run', return_value=gpu), patch('benchmark.monitor.get_url', return_value='gauge 4\n'):
                monitor.phase = 'steady'
                for i in range(30):
                    monitor.sample(i)
                self.assertEqual(len(list(Path(directory).iterdir())), 3)
                monitor.phase = 'profile'
                monitor.sample(30)
                self.assertEqual(len(list(Path(directory).iterdir())), 5)
            self.assertEqual(monitor.store.count, 31)
            self.assertFalse((Path(directory)/'system_samples.jsonl').exists())

    def test_transition_epoch_detects_switch_away_and_back(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(directory, 'offline://single', '1')
            monitor.phase = 'steady'
            def metrics(*args, **kwargs):
                monitor.phase = 'retry'
                monitor.phase = 'steady'
                return 'x 1\n'
            snapshot = dict(server_process_cpu_percent=None, server_process_rss_bytes=None)
            with patch('benchmark.monitor.process_snapshot', return_value=(snapshot,None)), patch('benchmark.monitor.subprocess.run', side_effect=OSError('no GPU')), patch('benchmark.monitor.get_url', side_effect=metrics):
                monitor.sample(0)
            self.assertEqual(monitor.store.phases, [('steady','transition')])

    def test_clean_shutdown_writes_one_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(directory, 'offline://single', '1', interval=.01)
            def sample(index):
                monitor.store.append(record(index), {('x',''): index})
                monitor.stop_event.set()
            with patch.object(monitor, 'sample', side_effect=sample):
                with monitor:
                    pass
            self.assertEqual(ScalarArchive(Path(directory)/'scalars.zip').metadata['count'], 1)
            self.assertTrue(json.loads((Path(directory)/'monitor_summary.json').read_text())['complete'])

    def test_fixed_deadline_skips_slow_round_without_catchup(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(directory, '', '1', interval=1)
            clock = [0.]
            waits = []
            def sample(index):
                clock[0] += 2.4 if index == 0 else .2
                if index == 1:
                    monitor.stop_event.set()
            def wait(delay):
                waits.append(delay)
                clock[0] += delay
            with patch.object(monitor, 'sample', side_effect=sample), patch('benchmark.monitor.time.monotonic', side_effect=lambda: clock[0]), patch.object(monitor.stop_event, 'wait', side_effect=wait):
                monitor._run()
            self.assertAlmostEqual(waits[0], .6)
            self.assertAlmostEqual(waits[1], .8)

    def test_archive_resource_statistics_and_optional_dcgm(self):
        store = ScalarStore()
        for i in range(3):
            store.append(record(i, value=i), {}, {('DCGM_OTHER','{gpu="1"}'): 2*i})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scalars.zip'
            store.save(path)
            archive = ScalarArchive(path)
            actual = dict(archive.statistics('system', 0, 4))
            self.assertEqual(actual[('gpu_utilization_percent','')]['p95'], 1.9)
            self.assertEqual(actual[('pinned_memory_bytes','')]['missing'], 3)
            self.assertEqual(list(archive.column(('dcgm','DCGM_OTHER','{gpu="1"}'))), [0,2,4])

    def test_empty_archive_is_valid_missing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scalars.zip'
            ScalarStore().save(path)
            archive = ScalarArchive(path)
            self.assertEqual(dict(archive.statistics('system', 0, 1)), {})
            self.assertEqual(dict(archive.prometheus_statistics(0, 1)), {})

    def test_invalid_interval_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for interval in [0,-1,float('nan'),float('inf')]:
                with self.assertRaises(ValueError):
                    Monitor(directory, '', '1', interval=interval)


if __name__ == '__main__':
    unittest.main()
