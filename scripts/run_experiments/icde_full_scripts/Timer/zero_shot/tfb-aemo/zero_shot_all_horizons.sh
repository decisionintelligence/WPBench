#!/usr/bin/env bash
# Generated portable ICDE missing-all-sheets task. Source: scripts/hpo/icde_day5_cleanup_longtail_20260516/tasks/tfb-aemo/Timer_zero_shot_eval.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/rolling_forecast_config.json}"
FINAL_CONFIG_PATH="${FINAL_CONFIG_PATH:-config/rolling_forecast_config_report_dtw.json}"
BENCHMARK_CONFIG_PATH="${FINAL_CONFIG_PATH##*/}"
SAVE_ROOT="${SAVE_ROOT:-result/icde_missing_all_sheets_week_20260523/manual}"
N_TRIALS="${N_TRIALS:-2}"
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-nmse}"
SAMPLER="${SAMPLER:-tpe}"
SEED="${SEED:-2021}"
SEQ_LEN="${SEQ_LEN:-576}"
GPUS="${GPUS:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-1}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-60000}"
RUN_BACKFILL_AFTER_HPO="${RUN_BACKFILL_AFTER_HPO:-true}"
BACKFILL_CONFIG_PATH="${BACKFILL_CONFIG_PATH:-config/rolling_forecast_config_report_dtw.json}"
BACKFILL_INCLUDE_DTW="${BACKFILL_INCLUDE_DTW:-true}"
FINAL_INCLUDE_DTW="${FINAL_INCLUDE_DTW:-true}"
DATA_NAME=forecasting_wp1_all_shapes_v1/tfb-aemo.csv
DATASET_STEM=tfb-aemo

GPU_ARGS=()
if [[ -n "${GPUS}" ]]; then
  read -r -a GPU_IDS <<< "${GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

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

echo "[EVAL] dataset=${DATA_NAME} model=Timer mode=zero_shot horizon=12"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":12,"stride":12,"target_channel":[0]}' \
  --model-name tsfm.Timer \
  --adapter timer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"horizon\":12,\"pred_len\":12,\"shot_mode\":\"zero_shot\"}" \
  "${FINAL_METRICS_ARGS[@]}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Timer/zero_shot/${DATASET_STEM}/horizon_12"

echo "[EVAL] dataset=${DATA_NAME} model=Timer mode=zero_shot horizon=24"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":24,"stride":24,"target_channel":[0]}' \
  --model-name tsfm.Timer \
  --adapter timer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"horizon\":24,\"pred_len\":24,\"shot_mode\":\"zero_shot\"}" \
  "${FINAL_METRICS_ARGS[@]}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Timer/zero_shot/${DATASET_STEM}/horizon_24"

echo "[EVAL] dataset=${DATA_NAME} model=Timer mode=zero_shot horizon=72"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":72,"stride":72,"target_channel":[0]}' \
  --model-name tsfm.Timer \
  --adapter timer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"horizon\":72,\"pred_len\":72,\"shot_mode\":\"zero_shot\"}" \
  "${FINAL_METRICS_ARGS[@]}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Timer/zero_shot/${DATASET_STEM}/horizon_72"

echo "[EVAL] dataset=${DATA_NAME} model=Timer mode=zero_shot horizon=144"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":144,"stride":144,"target_channel":[0]}' \
  --model-name tsfm.Timer \
  --adapter timer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"horizon\":144,\"pred_len\":144,\"shot_mode\":\"zero_shot\"}" \
  "${FINAL_METRICS_ARGS[@]}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Timer/zero_shot/${DATASET_STEM}/horizon_144"
