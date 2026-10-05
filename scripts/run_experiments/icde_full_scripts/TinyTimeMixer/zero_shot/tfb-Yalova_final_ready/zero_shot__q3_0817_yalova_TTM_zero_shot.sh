#!/usr/bin/env bash
set -euo pipefail

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/rolling_forecast_config_report_dtw.json}"
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
DATA_NAME=forecasting_wp1_batch00_smoke/tfb-Yalova_final_ready.csv
DATASET_STEM=tfb-Yalova_final_ready

GPU_ARGS=()
if [[ -n "${GPUS}" ]]; then
  read -r -a GPU_IDS <<< "${GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

echo "[EVAL] dataset=${DATA_NAME} model=TinyTimeMixer mode=zero_shot horizon=12"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":12,"stride":12,"target_channel":[0]}' \
  --model-name tsfm.TinyTimeMixer \
  --adapter tinytimemixer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"finetune_style\":\"benchmark\",\"horizon\":48,\"norm\":true,\"pred_len\":48,\"prefix_horizon_slice\":true,\"shot_mode\":\"zero_shot\",\"trainer_num_workers\":0,\"trainer_seed\":2021,\"ttm_checkpoint_path\":\"checkpoints/tsfm/ttm_local/512-48-ft-r2.1\",\"ttm_context_length\":512}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/TinyTimeMixer/zero_shot/${DATASET_STEM}/horizon_12"

echo "[EVAL] dataset=${DATA_NAME} model=TinyTimeMixer mode=zero_shot horizon=24"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":24,"stride":24,"target_channel":[0]}' \
  --model-name tsfm.TinyTimeMixer \
  --adapter tinytimemixer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"finetune_style\":\"benchmark\",\"horizon\":48,\"norm\":true,\"pred_len\":48,\"prefix_horizon_slice\":true,\"shot_mode\":\"zero_shot\",\"trainer_num_workers\":0,\"trainer_seed\":2021,\"ttm_checkpoint_path\":\"checkpoints/tsfm/ttm_local/512-48-ft-r2.1\",\"ttm_context_length\":512}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/TinyTimeMixer/zero_shot/${DATASET_STEM}/horizon_24"

echo "[EVAL] dataset=${DATA_NAME} model=TinyTimeMixer mode=zero_shot horizon=72"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":72,"stride":72,"target_channel":[0]}' \
  --model-name tsfm.TinyTimeMixer \
  --adapter tinytimemixer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"finetune_style\":\"benchmark\",\"horizon\":96,\"norm\":true,\"pred_len\":96,\"prefix_horizon_slice\":true,\"shot_mode\":\"zero_shot\",\"trainer_num_workers\":0,\"trainer_seed\":2021,\"ttm_checkpoint_path\":\"checkpoints/tsfm/ttm_local/512-96-ft-r2.1\",\"ttm_context_length\":512}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/TinyTimeMixer/zero_shot/${DATASET_STEM}/horizon_72"

echo "[EVAL] dataset=${DATA_NAME} model=TinyTimeMixer mode=zero_shot horizon=144"
"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path "${BENCHMARK_CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --strategy-args '{"deterministic":"efficient","horizon":144,"stride":144,"target_channel":[0]}' \
  --model-name tsfm.TinyTimeMixer \
  --adapter tinytimemixer_adapter \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"finetune_style\":\"benchmark\",\"horizon\":192,\"norm\":true,\"pred_len\":192,\"prefix_horizon_slice\":true,\"shot_mode\":\"zero_shot\",\"trainer_num_workers\":0,\"trainer_seed\":2021,\"ttm_checkpoint_path\":\"checkpoints/tsfm/ttm_local/512-192-r2\",\"ttm_context_length\":512}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/TinyTimeMixer/zero_shot/${DATASET_STEM}/horizon_144"
