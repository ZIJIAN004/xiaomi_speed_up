"""Compact scalar sampling: RAM during inference, one portable archive at shutdown."""
from array import array
import json
import math
import sys
import zipfile

from benchmark.metrics import dump_json


class ScalarStore:
    def __init__(self):
        self.columns = {}
        self.count = 0
        self.phases = []
        self.online = {}
        self.phase_counts = {}

    def append(self, record, prometheus, dcgm=None):
        phase = (record['phase'], record['phase_end'])
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        if phase not in self.phases:
            self.phases.append(phase)
        values = {('system', key, ''): value for key, value in record.items()
                  if value is None or (isinstance(value, (float, int)) and not isinstance(value, bool))}
        values.update({('memory', key, ''): value for key, value in record.get('system_memory', {}).items()})
        values.update({('prometheus', name, labels): value for (name, labels), value in prometheus.items()})
        values.update({('dcgm', name, labels): value for (name, labels), value in (dcgm or {}).items()})
        values[('system', 'phase_id', '')] = self.phases.index(phase)
        for key in values.keys() - self.columns.keys():
            self.columns[key] = array('d', [math.nan]) * self.count
        for key, column in self.columns.items():
            value = values.get(key)
            value = float(value) if value is not None else math.nan
            column.append(value)
            if math.isfinite(value):
                # Welford updates; no sorting or historical scans in the sampling loop.
                state = self.online.setdefault((phase, key), [0, 0., 0., value, value, None])
                state[0] += 1
                delta = value - state[1]
                state[1] += delta / state[0]
                state[2] += delta * (value - state[1])
                state[3] = min(state[3], value)
                if state[5] is None or value > state[4]:
                    state[4], state[5] = value, record['monotonic_s']
        self.count += 1

    def checkpoint(self, path, interval, errors, complete=False):
        rows = []
        for (phase, key), (count, mean, m2, low, high, peak_time) in self.online.items():
            rows.append(dict(phase=phase, series=key, count=count, mean=mean,
                             std=math.sqrt(max(0., m2/count)), min=low, max=high,
                             max_monotonic_s=peak_time, missing=self.phase_counts[phase]-count))
        known = {(tuple(row['phase']), tuple(row['series'])) for row in rows}
        for phase, total in self.phase_counts.items():
            for key in self.columns:
                if (phase, key) not in known:
                    rows.append(dict(phase=phase, series=key, count=0, missing=total,
                                     mean=None, std=None, min=None, max=None, max_monotonic_s=None))
        temporary = path.with_suffix('.tmp')
        dump_json(temporary, dict(sample_count=self.count, nominal_interval_s=interval,
                                 complete=complete, statistics=rows, errors=errors,
                                 phases=[dict(phase=phase, count=count) for phase, count in self.phase_counts.items()],
                                 quantiles='exact sample quantiles available from scalars.zip after clean shutdown; not time-weighted',
                                 crash_scope='online aggregates only; unarchived scalar arrays are lost on process crash'))
        temporary.replace(path)

    def save(self, path):
        temporary = path.with_suffix('.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr('metadata.json', json.dumps(dict(count=self.count, phases=self.phases,
                              columns=list(self.columns), byteorder=sys.byteorder)))
            for index, column in enumerate(self.columns.values()):
                archive.writestr(str(index), memoryview(column).cast('B'))
        temporary.replace(path)


class ScalarArchive:
    """Read columns independently; never expand all Prometheus samples into dictionaries."""
    def __init__(self, path):
        self.path = path
        with zipfile.ZipFile(path) as archive:
            self.metadata = json.loads(archive.read('metadata.json'))
        self.keys = [tuple(key) for key in self.metadata['columns']]

    def column(self, key):
        with zipfile.ZipFile(self.path) as archive:
            result = array('d')
            result.frombytes(archive.read(str(self.keys.index(key))))
        if self.metadata['byteorder'] != sys.byteorder:
            result.byteswap()
        return result

    def system_records(self):
        columns = {key[1]: self.column(key) for key in self.keys if key[0] == 'system'}
        for index in range(self.metadata['count']):
            record = {key: column[index] if math.isfinite(column[index]) else None
                      for key, column in columns.items()}
            phase = self.metadata['phases'][int(record.pop('phase_id'))]
            record.update(phase=phase[0], phase_end=phase[1])
            yield record

    def steady_indices(self, start, end):
        if not self.metadata['count']:
            return []
        starts = self.column(('system', 'monotonic_s', ''))
        ends = self.column(('system', 'sample_end_monotonic_s', ''))
        phases = self.column(('system', 'phase_id', ''))
        return [i for i in range(self.metadata['count'])
                if self.metadata['phases'][int(phases[i])] == ['steady', 'steady']
                and start <= starts[i] <= ends[i] <= end]

    def statistics(self, group, start, end):
        from benchmark.metrics import stats
        indices = self.steady_indices(start, end)
        for key in self.keys:
            if key[0] == group:
                column = self.column(key)
                summary = stats(column[i] for i in indices)
                if group == 'system':
                    summary['missing'] = len(indices) - summary['count']
                yield key[1:], summary

    def prometheus_statistics(self, start, end):
        return self.statistics('prometheus', start, end)
