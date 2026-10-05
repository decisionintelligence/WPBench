#!/usr/bin/env bash
set -euo pipefail

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-rolling_forecast_config_report_dtw.json}"
SAVE_ROOT="${SAVE_ROOT:-result/hpo_green_fix4_icde_day1_metric_repair_20260515}"
GPUS="${GPUS:?GPUS must be set by the lane runner}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-1}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-60000}"
NUM_ROLLINGS="${NUM_ROLLINGS:-48000}"
DATASET_STEM='tfb-UEPS_panel_wide_filled__MTMV_reduced_3V_active_power_total'
DATA_NAME="forecasting_wp1_all_shapes_v1/${DATASET_STEM}.csv"

echo "[BENCH] model=LR dataset=${DATASET_STEM} horizon=12 gpu=${GPUS}"
strategy_args="{\"strategy_name\":\"rolling_forecast\",\"horizon\":12,\"stride\":12,\"num_rollings\":${NUM_ROLLINGS},\"target_channel\":[0],\"save_true_pred\":false}"
"${PYTHON_BIN}" ./scripts/run_benchmark.py \
    --config-path "${CONFIG_PATH}" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "${strategy_args}" \
    --model-name "darts.LinearRegressionModel" \
    --metrics '{"name":"mae"}' \
    '{"name":"mse"}' \
    '{"name":"rmse"}' \
    '{"name":"nmae"}' \
    '{"name":"nmse"}' \
    '{"name":"nrmse"}' \
    '{"name":"nmbe"}' \
    '{"name":"maape"}' \
    '{"name":"mape"}' \
    '{"name":"smape"}' \
    '{"name":"mase"}' \
    '{"name":"rmsse"}' \
    '{"name":"wape"}' \
    '{"name":"msmape"}' \
    '{"name":"edf"}' \
    '{"name":"opencity_mae"}' \
    '{"name":"opencity_mse"}' \
    '{"name":"opencity_rmse"}' \
    '{"name":"mae_norm"}' \
    '{"name":"mse_norm"}' \
    '{"name":"rmse_norm"}' \
    '{"name":"mape_norm"}' \
    '{"name":"smape_norm"}' \
    '{"name":"mase_norm"}' \
    '{"name":"wape_norm"}' \
    '{"name":"msmape_norm"}' \
    '{"name":"dtw"}' \
    '{"name":"ndtw"}' \
    '{"name":"nddtw"}' \
    --eval-backend sequential \
    --num-workers "${NUM_WORKERS}" \
    --num-cpus "${NUM_CPUS}" \
    --gpus "${GPUS}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --save-path "${SAVE_ROOT}/LR/${DATASET_STEM}/horizon_12"

echo "[BENCH] model=LR dataset=${DATASET_STEM} horizon=24 gpu=${GPUS}"
strategy_args="{\"strategy_name\":\"rolling_forecast\",\"horizon\":24,\"stride\":24,\"num_rollings\":${NUM_ROLLINGS},\"target_channel\":[0],\"save_true_pred\":false}"
"${PYTHON_BIN}" ./scripts/run_benchmark.py \
    --config-path "${CONFIG_PATH}" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "${strategy_args}" \
    --model-name "darts.LinearRegressionModel" \
    --metrics '{"name":"mae"}' \
    '{"name":"mse"}' \
    '{"name":"rmse"}' \
    '{"name":"nmae"}' \
    '{"name":"nmse"}' \
    '{"name":"nrmse"}' \
    '{"name":"nmbe"}' \
    '{"name":"maape"}' \
    '{"name":"mape"}' \
    '{"name":"smape"}' \
    '{"name":"mase"}' \
    '{"name":"rmsse"}' \
    '{"name":"wape"}' \
    '{"name":"msmape"}' \
    '{"name":"edf"}' \
    '{"name":"opencity_mae"}' \
    '{"name":"opencity_mse"}' \
    '{"name":"opencity_rmse"}' \
    '{"name":"mae_norm"}' \
    '{"name":"mse_norm"}' \
    '{"name":"rmse_norm"}' \
    '{"name":"mape_norm"}' \
    '{"name":"smape_norm"}' \
    '{"name":"mase_norm"}' \
    '{"name":"wape_norm"}' \
    '{"name":"msmape_norm"}' \
    '{"name":"dtw"}' \
    '{"name":"ndtw"}' \
    '{"name":"nddtw"}' \
    --eval-backend sequential \
    --num-workers "${NUM_WORKERS}" \
    --num-cpus "${NUM_CPUS}" \
    --gpus "${GPUS}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --save-path "${SAVE_ROOT}/LR/${DATASET_STEM}/horizon_24"

echo "[BENCH] model=LR dataset=${DATASET_STEM} horizon=72 gpu=${GPUS}"
strategy_args="{\"strategy_name\":\"rolling_forecast\",\"horizon\":72,\"stride\":72,\"num_rollings\":${NUM_ROLLINGS},\"target_channel\":[0],\"save_true_pred\":false}"
"${PYTHON_BIN}" ./scripts/run_benchmark.py \
    --config-path "${CONFIG_PATH}" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "${strategy_args}" \
    --model-name "darts.LinearRegressionModel" \
    --metrics '{"name":"mae"}' \
    '{"name":"mse"}' \
    '{"name":"rmse"}' \
    '{"name":"nmae"}' \
    '{"name":"nmse"}' \
    '{"name":"nrmse"}' \
    '{"name":"nmbe"}' \
    '{"name":"maape"}' \
    '{"name":"mape"}' \
    '{"name":"smape"}' \
    '{"name":"mase"}' \
    '{"name":"rmsse"}' \
    '{"name":"wape"}' \
    '{"name":"msmape"}' \
    '{"name":"edf"}' \
    '{"name":"opencity_mae"}' \
    '{"name":"opencity_mse"}' \
    '{"name":"opencity_rmse"}' \
    '{"name":"mae_norm"}' \
    '{"name":"mse_norm"}' \
    '{"name":"rmse_norm"}' \
    '{"name":"mape_norm"}' \
    '{"name":"smape_norm"}' \
    '{"name":"mase_norm"}' \
    '{"name":"wape_norm"}' \
    '{"name":"msmape_norm"}' \
    '{"name":"dtw"}' \
    '{"name":"ndtw"}' \
    '{"name":"nddtw"}' \
    --eval-backend sequential \
    --num-workers "${NUM_WORKERS}" \
    --num-cpus "${NUM_CPUS}" \
    --gpus "${GPUS}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --save-path "${SAVE_ROOT}/LR/${DATASET_STEM}/horizon_72"

echo "[BENCH] model=LR dataset=${DATASET_STEM} horizon=144 gpu=${GPUS}"
strategy_args="{\"strategy_name\":\"rolling_forecast\",\"horizon\":144,\"stride\":144,\"num_rollings\":${NUM_ROLLINGS},\"target_channel\":[0],\"save_true_pred\":false}"
"${PYTHON_BIN}" ./scripts/run_benchmark.py \
    --config-path "${CONFIG_PATH}" \
    --data-name-list "${DATA_NAME}" \
    --strategy-args "${strategy_args}" \
    --model-name "darts.LinearRegressionModel" \
    --metrics '{"name":"mae"}' \
    '{"name":"mse"}' \
    '{"name":"rmse"}' \
    '{"name":"nmae"}' \
    '{"name":"nmse"}' \
    '{"name":"nrmse"}' \
    '{"name":"nmbe"}' \
    '{"name":"maape"}' \
    '{"name":"mape"}' \
    '{"name":"smape"}' \
    '{"name":"mase"}' \
    '{"name":"rmsse"}' \
    '{"name":"wape"}' \
    '{"name":"msmape"}' \
    '{"name":"edf"}' \
    '{"name":"opencity_mae"}' \
    '{"name":"opencity_mse"}' \
    '{"name":"opencity_rmse"}' \
    '{"name":"mae_norm"}' \
    '{"name":"mse_norm"}' \
    '{"name":"rmse_norm"}' \
    '{"name":"mape_norm"}' \
    '{"name":"smape_norm"}' \
    '{"name":"mase_norm"}' \
    '{"name":"wape_norm"}' \
    '{"name":"msmape_norm"}' \
    '{"name":"dtw"}' \
    '{"name":"ndtw"}' \
    '{"name":"nddtw"}' \
    --eval-backend sequential \
    --num-workers "${NUM_WORKERS}" \
    --num-cpus "${NUM_CPUS}" \
    --gpus "${GPUS}" \
    --timeout "${BENCHMARK_TIMEOUT}" \
    --save-path "${SAVE_ROOT}/LR/${DATASET_STEM}/horizon_144"
