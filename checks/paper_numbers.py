import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--reference', action='store_true', help='use reference data if fresh output is missing')
args = parser.parse_args()


def read_value(spec):
    experiment, filename = spec['file'].split('/', 1)
    directories = {
        **{name: ROOT / 'experiments/siso_main' / name / 'results' for name in ('E1', 'E2', 'E3', 'E4', 'B')},
        'e0_fixed_point': ROOT / 'experiments/e0_fixed_point/E0/results',
        'table1_extrema': ROOT / 'experiments/table1_extrema/results',
        'rician_and_checks': ROOT / 'experiments/rician_and_checks/results',
        'estimator_mc': ROOT / 'experiments/estimator_mc/results',
        'frozen_tx': ROOT / 'experiments/frozen_tx/results',
    }
    path = directories[experiment] / filename
    if not path.is_file() and args.reference:
        path = ROOT / 'reference_results' / spec['file']
        if not path.is_file() and filename == 'all_runs.json' and experiment in ('E1', 'E2', 'E3', 'E4'):
            with (ROOT / 'reference_results/figures' / f'{experiment}_data.csv').open() as source:
                data = [dict(nr=int(row['nr']), method=row['method'], geometry=dict(spectrum=row['spectrum'], displacement_lambda=float(row['d']), rho=dict(real=float(row['rho_real']), imag=float(row['rho_imag']))), service=dict(accuracy=float(row['accuracy']))) for row in csv.DictReader(source)]
            return data
    data = json.loads(path.read_text())
    for key in spec['key_path']:
        if isinstance(key, dict):
            data = [item for item in data if all(
                abs(float(item[k]) - float(v)) < 1e-6 if isinstance(v, (int, float)) else item[k] == v
                for k, v in key.items())]
        elif key == '*':
            continue
        elif isinstance(data, list) and isinstance(key, str):
            data = [item[key] for item in data]
        else:
            data = data[key]
    return data


passed = failed = 0
for row in json.loads((ROOT / 'checks/paper_values.json').read_text()):
    try:
        value = read_value(row)
        if 'formula' in row:
            value = eval(row['formula'], {'__builtins__': {}, 'x': value, 'mean': statistics.mean, 'min': min, 'max': max, 'abs': abs, 'math': math, 'next': next, 'all': all, 'int': int, 'j0': j0})
        values = value if isinstance(value, list) else [value]
        rounded = [round(float(item), digits) for item, digits in zip(values, row['rounding'])]
        ok = len(values) == len(row['printed_value']) and rounded == row['printed_value']
        rendered = ', '.join(f'{item:.{digits}f}' for item, digits in zip(rounded, row['rounding']))
        print(f"{'PASS' if ok else 'FAIL'} {row['id']}: paper {row['paper']}; computed {rendered}")
        passed += ok
        failed += not ok
    except (OSError, KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as error:
        print(f"FAIL {row['id']}: paper {row['paper']}; {error}")
        failed += 1
print(f'Paper summary: {passed} PASS, {failed} FAIL')
raise SystemExit(bool(failed))
