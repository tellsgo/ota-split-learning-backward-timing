#!/bin/bash
set -u
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=${PYTHON:-python3}
run_one() {
  case "$1" in
    E1|E2|E3|E4|B) (cd experiments/siso_main && "$PYTHON" -B -u "$1/src/launch.py") ;;
    e0_fixed_point) "$PYTHON" -B -u experiments/e0_fixed_point/E0/src/launch.py ;;
    table1_extrema) mkdir -p experiments/table1_extrema/figures/cache; : > experiments/table1_extrema/results/runs.jsonl; (cd experiments/table1_extrema && "$PYTHON" src/run_g02.py) ;;
    rician_and_checks) mkdir -p experiments/rician_and_checks/{results,logs,tmp}; : > experiments/rician_and_checks/results/runs.jsonl; "$PYTHON" experiments/rician_and_checks/experiment.py B && "$PYTHON" experiments/rician_and_checks/closeout.py C ;;
    estimator_mc) mkdir -p experiments/estimator_mc/results; "$PYTHON" experiments/estimator_mc/minimal_trial.py ;;
    frozen_tx) "$PYTHON" -B -u experiments/frozen_tx/src/frozen_tx.py ;;
    *) echo "Unknown target: $1" >&2; return 2 ;;
  esac
}
case "${1:-}" in
  quick)
    pids=()
    for target in E1 E2 E3 E4 B e0_fixed_point table1_extrema rician_and_checks estimator_mc frozen_tx; do
      run_one "$target" & pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    exit "$status" ;;
  check) "$PYTHON" checks/compare_reference.py ;;
  paper) "$PYTHON" checks/paper_numbers.py "${@:2}" ;;
  figures) "$PYTHON" figures/make_figures.py ;;
  *) run_one "${1:-}" ;;
esac
