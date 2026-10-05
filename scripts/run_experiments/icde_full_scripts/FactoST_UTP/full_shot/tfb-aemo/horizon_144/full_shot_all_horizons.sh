#!/usr/bin/env bash
set -euo pipefail

cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-config/rolling_forecast_config.json}"
BENCHMARK_CONFIG_PATH="${CONFIG_PATH##*/}"
SAVE_ROOT="${SAVE_ROOT:-result/hpo_all_v4_wp1_all_shapes_fix4_dtw_backfill}"
N_TRIALS="${N_TRIALS:-2}"
OBJECTIVE_METRIC="${OBJECTIVE_METRIC:-nmse}"
SAMPLER="${SAMPLER:-tpe}"
SEED="${SEED:-2021}"
SEQ_LEN="${SEQ_LEN:-576}"
GPUS="${GPUS:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
NUM_CPUS="${NUM_CPUS:-1}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-60000}"
RUN_BACKFILL_AFTER_HPO="${RUN_BACKFILL_AFTER_HPO:-false}"
BACKFILL_CONFIG_PATH="${BACKFILL_CONFIG_PATH:-config/rolling_forecast_config_report_dtw.json}"
BACKFILL_INCLUDE_DTW="${BACKFILL_INCLUDE_DTW:-true}"
DATA_NAME=forecasting_wp1_all_shapes_v1/tfb-aemo.csv
DATASET_STEM=tfb-aemo

GPU_ARGS=()
if [[ -n "${GPUS}" ]]; then
  read -r -a GPU_IDS <<< "${GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

echo "[HPO] dataset=${DATA_NAME} model=FactoST_UTP mode=full_shot horizon=12"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --adapter factost_utp_finetune_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":12,"stride":12,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"factost_d_ff\":1024,\"factost_d_model\":256,\"factost_dropout\":0.2,\"factost_n_heads\":4,\"factost_n_layers\":3,\"factost_num_token\":3,\"factost_revin\":true,\"loss\":\"MAE\",\"lr_type\":\"OneCycle\",\"norm\":false,\"patch_len\":16,\"pretrain_model_path\":\"checkpoints/stfm/factost_utp2_tiny_4utp.pt\",\"shot_mode\":\"full_shot\",\"stride\":16}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 12 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/FactoST_UTP/full_shot/${DATASET_STEM}/horizon_12" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=FactoST_UTP mode=full_shot horizon=24"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --adapter factost_utp_finetune_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":24,"stride":24,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"factost_d_ff\":1024,\"factost_d_model\":256,\"factost_dropout\":0.2,\"factost_n_heads\":4,\"factost_n_layers\":3,\"factost_num_token\":3,\"factost_revin\":true,\"loss\":\"MAE\",\"lr_type\":\"OneCycle\",\"norm\":false,\"patch_len\":16,\"pretrain_model_path\":\"checkpoints/stfm/factost_utp2_tiny_4utp.pt\",\"shot_mode\":\"full_shot\",\"stride\":16}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 24 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/FactoST_UTP/full_shot/${DATASET_STEM}/horizon_24" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=FactoST_UTP mode=full_shot horizon=72"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --adapter factost_utp_finetune_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":72,"stride":72,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"factost_d_ff\":1024,\"factost_d_model\":256,\"factost_dropout\":0.2,\"factost_n_heads\":4,\"factost_n_layers\":3,\"factost_num_token\":3,\"factost_revin\":true,\"loss\":\"MAE\",\"lr_type\":\"OneCycle\",\"norm\":false,\"patch_len\":16,\"pretrain_model_path\":\"checkpoints/stfm/factost_utp2_tiny_4utp.pt\",\"shot_mode\":\"full_shot\",\"stride\":16}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 72 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/FactoST_UTP/full_shot/${DATASET_STEM}/horizon_72" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"

echo "[HPO] dataset=${DATA_NAME} model=FactoST_UTP mode=full_shot horizon=144"
"${PYTHON_BIN}" scripts/run_hpo.py \
  --config-path "${CONFIG_PATH}" \
  --data-name-list "${DATA_NAME}" \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --adapter factost_utp_finetune_adapter \
  --strategy-args '{"deterministic":"efficient","horizon":144,"stride":144,"target_channel":[0]}' \
  --model-hyper-params "{\"seq_len\":${SEQ_LEN},\"factost_d_ff\":1024,\"factost_d_model\":256,\"factost_dropout\":0.2,\"factost_n_heads\":4,\"factost_n_layers\":3,\"factost_num_token\":3,\"factost_revin\":true,\"loss\":\"MAE\",\"lr_type\":\"OneCycle\",\"norm\":false,\"patch_len\":16,\"pretrain_model_path\":\"checkpoints/stfm/factost_utp2_tiny_4utp.pt\",\"shot_mode\":\"full_shot\",\"stride\":16}" \
  --n-trials "${N_TRIALS}" \
  --forecast-lengths 144 \
  --objective-metric "${OBJECTIVE_METRIC}" \
  --sampler "${SAMPLER}" \
  --seed "${SEED}" \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS}" \
  --num-cpus "${NUM_CPUS}" "${GPU_ARGS[@]}" \
  --benchmark-timeout "${BENCHMARK_TIMEOUT}" \
  --save-path "${SAVE_ROOT}/FactoST_UTP/full_shot/${DATASET_STEM}/horizon_144" \
  --run-backfill-after-hpo "${RUN_BACKFILL_AFTER_HPO}" \
  --backfill-config-path "${BACKFILL_CONFIG_PATH}" \
  --backfill-include-dtw "${BACKFILL_INCLUDE_DTW}"
