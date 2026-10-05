#!/usr/bin/env bash
set -euo pipefail

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/rolling_forecast_config.json}"
BENCHMARK_CONFIG_PATH="${CONFIG_PATH##*/}"
SAVE_ROOT="${SAVE_ROOT:-result/hpo_green_fix4_icde_day4_foundation_models_20260516}"
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
DATA_NAME=forecasting_wp1_all_shapes_v1/tfb-maelstrom_turbine_3_univariate.csv
DATASET_STEM=tfb-maelstrom_turbine_3_univariate

GPU_ARGS=()
if [[ -n "${GPUS}" ]]; then
  read -r -a GPU_IDS <<< "${GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "[HPO] dataset=${DATA_NAME} model=Toto mode=few_shot_10pct horizon=12"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name tsfm.Toto \
  --adapter toto_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":12,"stride":12,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":1,\"checkpoint_path\":\"checkpoints/tsfm/Toto-Open-Base-1.0\",\"few_shot_ratio\":0.1,\"finetune_style\":\"official_trainer\",\"norm\":false,\"num_epochs\":1,\"num_samples\":32,\"num_train_samples\":32,\"samples_per_batch\":32,\"seed\":2021,\"shot_mode\":\"few_shot\",\"toto_num_train_samples\":32,\"toto_official_num_train_samples\":32,\"use_kv_cache\":true}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 12 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Toto/few_shot_10pct/${DATASET_STEM}/horizon_12" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Toto mode=few_shot_10pct horizon=24"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name tsfm.Toto \
  --adapter toto_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":24,"stride":24,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":1,\"checkpoint_path\":\"checkpoints/tsfm/Toto-Open-Base-1.0\",\"few_shot_ratio\":0.1,\"finetune_style\":\"official_trainer\",\"norm\":false,\"num_epochs\":1,\"num_samples\":32,\"num_train_samples\":32,\"samples_per_batch\":32,\"seed\":2021,\"shot_mode\":\"few_shot\",\"toto_num_train_samples\":32,\"toto_official_num_train_samples\":32,\"use_kv_cache\":true}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 24 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Toto/few_shot_10pct/${DATASET_STEM}/horizon_24" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Toto mode=few_shot_10pct horizon=72"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name tsfm.Toto \
  --adapter toto_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":72,"stride":72,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":1,\"checkpoint_path\":\"checkpoints/tsfm/Toto-Open-Base-1.0\",\"few_shot_ratio\":0.1,\"finetune_style\":\"official_trainer\",\"norm\":false,\"num_epochs\":1,\"num_samples\":32,\"num_train_samples\":32,\"samples_per_batch\":32,\"seed\":2021,\"shot_mode\":\"few_shot\",\"toto_num_train_samples\":32,\"toto_official_num_train_samples\":32,\"use_kv_cache\":true}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 72 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Toto/few_shot_10pct/${DATASET_STEM}/horizon_72" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=Toto mode=few_shot_10pct horizon=144"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name tsfm.Toto \
  --adapter toto_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":144,"stride":144,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"batch_size\":1,\"checkpoint_path\":\"checkpoints/tsfm/Toto-Open-Base-1.0\",\"few_shot_ratio\":0.1,\"finetune_style\":\"official_trainer\",\"norm\":false,\"num_epochs\":1,\"num_samples\":32,\"num_train_samples\":32,\"samples_per_batch\":32,\"seed\":2021,\"shot_mode\":\"few_shot\",\"toto_num_train_samples\":32,\"toto_official_num_train_samples\":32,\"use_kv_cache\":true}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 144 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/Toto/few_shot_10pct/${DATASET_STEM}/horizon_144" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"
