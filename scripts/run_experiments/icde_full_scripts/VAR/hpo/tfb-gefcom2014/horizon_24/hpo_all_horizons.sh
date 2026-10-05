#!/usr/bin/env bash
# Archived per-dataset script reconstructed from /home/wpbench_51/scripts/hpo/remote_lr_var_backfill_20260514/manifest.csv.
# Original runner: scripts/hpo/remote_lr_var_backfill_20260514/run_var.sh
# This script preserves the historical LR/VAR backfill behavior that generated
# the filled metric cells in fixed_ICDE_online21.xlsx. It does not explicitly
# request dtw/ndtw/nddtw metrics; use Day1 metric repair scripts for DTW repair.
set +e

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-rolling_forecast_config_report_dtw.json}"
SAVE_ROOT="${SAVE_ROOT:-${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_fix4_remote_lr_var_backfill_20260514}"
LOG_ROOT="${LOG_ROOT:-${WPBENCH_LOG_ROOT:-/home/wpbench_51/wpbench_artifacts/logs}/hpo_green_fix4_remote_lr_var_backfill_20260514/gpu5_var}"
GPUS="${GPUS:-5}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-1}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-60000}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DATASET_STEM='tfb-gefcom2014'
HORIZONS=(12 24 72 144)
FAILED=0

mkdir -p "${LOG_ROOT}/${DATASET_STEM}"

has_report() {
  local save_path="$1"
  [[ -d "${save_path}" ]] && find "${save_path}" -name 'test_report*.csv' -print -quit | grep -q .
}

run_one() {
  local horizon="$1"
  local csv_path="${WPBENCH_FORECASTING_DATASET_PATH:-/home/wpbench_51/wpbench_artifacts/data/forecasting}/forecasting_wp1_all_shapes_v1/${DATASET_STEM}.csv"
  local data_name="forecasting_wp1_all_shapes_v1/${DATASET_STEM}.csv"
  local save_path="${SAVE_ROOT}/VAR/${DATASET_STEM}/horizon_${horizon}"
  local log="${LOG_ROOT}/${DATASET_STEM}/VAR_h${horizon}.log"
  local strategy_args

  if [[ ! -f "${csv_path}" ]]; then
    echo "[ERROR] $(date '+%F %T') model=VAR dataset=${DATASET_STEM} horizon=${horizon} reason=missing_csv path=${csv_path}"
    return 2
  fi
  if has_report "${save_path}"; then
    echo "[SKIP] $(date '+%F %T') model=VAR dataset=${DATASET_STEM} horizon=${horizon} reason=existing_test_report"
    return 0
  fi

  strategy_args=$(printf '{"strategy_name":"rolling_forecast","horizon":%s,"stride":%s,"num_rollings":%s,"target_channel":[0],"save_true_pred":false}' "${horizon}" "${horizon}" "${NUM_ROLLINGS}")

  echo "[START] $(date '+%F %T') model=VAR gpu=${GPUS} dataset=${DATASET_STEM} horizon=${horizon}"
  "${PYTHON_BIN}" ./scripts/run_benchmark.py \
    --config-path "${CONFIG_PATH}" \
    --data-name-list "${data_name}" \
    --strategy-args "${strategy_args}" \
    --model-name "self_impl.VAR_model" \
    --model-hyper-params '{"lags":13}' \
    --eval-backend sequential \
    --num-workers "${NUM_WORKERS}" \
    --num-cpus "${NUM_CPUS}" \
    --gpus "${GPUS}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --save-path "${save_path}" \
    > "${log}" 2>&1
  local code=$?
  echo "[END] $(date '+%F %T') model=VAR gpu=${GPUS} dataset=${DATASET_STEM} horizon=${horizon} exit=${code} log=${log}"
  return "${code}"
}

echo "[SCRIPT_START] $(date '+%F %T') model=VAR gpu=${GPUS} dataset=${DATASET_STEM} horizons=${HORIZONS[*]} save_root=${SAVE_ROOT}"
for horizon in "${HORIZONS[@]}"; do
  if ! run_one "${horizon}"; then
    FAILED=$((FAILED + 1))
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

echo "[SCRIPT_DONE] $(date '+%F %T') model=VAR dataset=${DATASET_STEM} failed=${FAILED}"
exit "${FAILED}"
