#!/usr/bin/env bash
set -euo pipefail

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/rolling_forecast_config.json}"
BENCHMARK_CONFIG_PATH="${CONFIG_PATH##*/}"
SAVE_ROOT="${SAVE_ROOT:-result/hpo_green_fix4_icde_day3_core_short_models_20260516}"
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
DATA_NAME=forecasting_wp1_all_shapes_v1/tfb-re_europe_wind_signal_ECMWF_long_no_null__MT1V_reduced_38T_1V_power.csv
DATASET_STEM=tfb-re_europe_wind_signal_ECMWF_long_no_null__MT1V_reduced_38T_1V_power

GPU_ARGS=()
if [[ -n "${GPUS}" ]]; then
  read -r -a GPU_IDS <<< "${GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

echo "[HPO] dataset=${DATA_NAME} model=Xlinear horizon=12"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name xlinear.Xlinear \
  --strategy-args '{"deterministic":"efficient","horizon":12,"stride":12,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":32,\"num_epochs\":10,\"num_workers\":0,\"patience\":3}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 12 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Xlinear/${DATASET_STEM}/horizon_12" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Xlinear horizon=24"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name xlinear.Xlinear \
  --strategy-args '{"deterministic":"efficient","horizon":24,"stride":24,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":32,\"num_epochs\":10,\"num_workers\":0,\"patience\":3}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 24 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Xlinear/${DATASET_STEM}/horizon_24" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Xlinear horizon=72"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name xlinear.Xlinear \
  --strategy-args '{"deterministic":"efficient","horizon":72,"stride":72,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":32,\"num_epochs\":10,\"num_workers\":0,\"patience\":3}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 72 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Xlinear/${DATASET_STEM}/horizon_72" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Xlinear horizon=144"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name xlinear.Xlinear \
  --strategy-args '{"deterministic":"efficient","horizon":144,"stride":144,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":32,\"num_epochs\":10,\"num_workers\":0,\"patience\":3}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 144 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Xlinear/${DATASET_STEM}/horizon_144" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"
