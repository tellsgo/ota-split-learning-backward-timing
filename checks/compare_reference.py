import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    **{name: ROOT / 'experiments' / 'siso_main' / name / 'results' for name in ('E1', 'E2', 'E3', 'E4', 'B')},
    'e0_fixed_point': ROOT / 'experiments/e0_fixed_point/E0/results',
    'table1_extrema': ROOT / 'experiments/table1_extrema/results',
    'rician_and_checks': ROOT / 'experiments/rician_and_checks/results',
    'estimator_mc': ROOT / 'experiments/estimator_mc/results',
    'frozen_tx': ROOT / 'experiments/frozen_tx/results',
}
METADATA = {'time', 'time_utc', 'timestamp', 'unix', 'started_epoch', 'finished_epoch', 'wall_seconds', 'cpu_seconds', 'host', 'hostname', 'pid', 'path', 'source', 'cwd', 'python'}


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()
                if key.lower() not in METADATA and not key.lower().endswith(('_path', '_paths', '_timestamp'))}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


failed = False
passed = skipped = 0
for experiment, directory in EXPERIMENTS.items():
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    for path in files:
        reference = ROOT / 'reference_results' / experiment / path.relative_to(directory)
        if not reference.is_file():
            print(f'SKIP (intermediate, no reference) {experiment}/{path.relative_to(directory)}')
            skipped += 1
            continue
        match = path.read_bytes() == reference.read_bytes()
        if not match and reference.is_file() and path.suffix == '.json':
            try:
                match = normalized(json.loads(path.read_text())) == normalized(json.loads(reference.read_text()))
            except (UnicodeError, ValueError):
                pass
        print(f"{'PASS' if match else 'FAIL'} {experiment}/{path.relative_to(directory)}" + ('' if reference.is_file() else ' (reference missing)'))
        failed |= not match
        passed += match
    reference_dir = ROOT / 'reference_results' / experiment
    for reference in sorted(path for path in reference_dir.rglob('*') if path.is_file()):
        relative = reference.relative_to(reference_dir)
        if experiment == 'frozen_tx' and str(relative) == 'selected_controls.json':
            fresh = ROOT / 'experiments/frozen_tx/reference/selected_controls.json'
            if fresh.is_file():
                match = normalized(json.loads(fresh.read_text())) == normalized(json.loads(reference.read_text()))
                print(f"{'PASS' if match else 'FAIL'} {experiment}/{relative}")
                failed |= not match
                passed += match
            else:
                print(f'FAIL {experiment}/{relative} (fresh output missing)')
                failed = True
        elif not (directory / relative).is_file():
            print(f'FAIL {experiment}/{relative} (fresh output missing)')
            failed = True
print(f'Summary: {passed} passed, {skipped} intermediates skipped; ' + ('FAIL' if failed else 'PASS'))
raise SystemExit(int(failed))
