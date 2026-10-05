#!/usr/bin/env bash
# Generated portable ICDE missing-all-sheets task. Source: wpbench_icde_script/tasks/tfb-Yalova_final_ready/VAR/hpo_all_horizons.sh
# Archived per-dataset script reconstructed from ${PROJECT_ROOT}/scripts/hpo/remote_lr_var_backfill_20260514/manifest.csv.
# Original runner: scripts/hpo/remote_lr_var_backfill_20260514/run_var.sh
# This script preserves the historical LR/VAR backfill behavior that generated
# the filled metric cells in fixed_ICDE_online21.xlsx. It does not explicitly
# request dtw/ndtw/nddtw metrics; use Day1 metric repair scripts for DTW repair.
set +e

cd ${PROJECT_ROOT} || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-rolling_forecast_config_report_dtw.json}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/result/hpo_green_fix4_remote_lr_var_backfill_20260514}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/hpo_green_fix4_remote_lr_var_backfill_20260514/gpu5_var}"
GPUS="${GPUS:-5}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-1}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-60000}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

FINAL_INCLUDE_DTW="${FINAL_INCLUDE_DTW:-true}"
FINAL_METRICS_ARGS=()
if [[ "${FINAL_INCLUDE_DTW}" == "true" ]]; then
  FINAL_METRICS_ARGS=(
  '{"name":"mae"}'
  '{"name":"mse"}'
  '{"name":"rmse"}'
  '{"name":"nmae"}'
  '{"name":"nmse"}'
  '{"name":"nrmse"}'
  '{"name":"nmbe"}'
  '{"name":"maape"}'
  '{"name":"mape"}'
  '{"name":"smape"}'
  '{"name":"mase"}'
  '{"name":"rmsse"}'
  '{"name":"wape"}'
  '{"name":"msmape"}'
  '{"name":"edf"}'
  '{"name":"opencity_mae"}'
  '{"name":"opencity_mse"}'
  '{"name":"opencity_rmse"}'
  '{"name":"mae_norm"}'
  '{"name":"mse_norm"}'
  '{"name":"rmse_norm"}'
  '{"name":"mape_norm"}'
  '{"name":"smape_norm"}'
  '{"name":"mase_norm"}'
  '{"name":"wape_norm"}'
  '{"name":"msmape_norm"}'
  '{"name":"dtw"}'
  '{"name":"ndtw"}'
  '{"name":"nddtw"}'
  )
  FINAL_METRICS_ARGS=(--metrics "${FINAL_METRICS_ARGS[@]}")
fi
DATASET_STEM=tfb-Yalova_final_ready
HORIZONS=(12 24 72 144)
FAILED=0

mkdir -p "${LOG_ROOT}/${DATASET_STEM}"

has_report() {
  local save_path="$1"
  local report=""
  [[ -d "${save_path}" ]] || return 1
  report=$(find "${save_path}" -name 'test_report*.csv' -print | sort | tail -1)
  [[ -n "${report}" ]] || return 1
  "${PYTHON_BIN}" -c 'import csv, sys; h=next(csv.reader(open(sys.argv[1], newline="")), []); sys.exit(0 if {"dtw","ndtw","nddtw"}.issubset(set(h)) else 1)' "${report}"
}

run_one() {
  local horizon="$1"
  local csv_path="${PROJECT_ROOT}/dataset/forecasting/forecasting_wp1_all_shapes_v1/${DATASET_STEM}.csv"
  local data_name="forecasting_wp1_all_shapes_v1/${DATASET_STEM}.csv"
  local save_path="${SAVE_ROOT}/VAR/${DATASET_STEM}/horizon_${horizon}"
  local log="${LOG_ROOT}/${DATASET_STEM}/VAR_h${horizon}.log"
  local strategy_args

  if [[ ! -f "${csv_path}" ]]; then
    echo "[ERROR] $(date '+%F %T') model=VAR dataset=${DATASET_STEM} horizon=${horizon} reason=missing_csv path=${csv_path}"
    return 2
  fi
  if has_report "${save_path}"; then
    echo "[SKIP] $(date '+%F %T') model=VAR dataset=${DATASET_STEM} horizon=${horizon} reason=existing_test_report_with_dtw"
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
    "${FINAL_METRICS_ARGS[@]}" \
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

DATA_NAME=forecasting_wp1_batch00_smoke/tfb-Yalova_final_ready.csv
