#!/usr/bin/env bash
set -euo pipefail

# One fixed-parameter experiment; no hyperparameter search.
WPBENCH_ROOT="${WPBENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)}"
export WPBENCH_ROOT
cd "${WPBENCH_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export WPBENCH_RESULT_ROOT="${WPBENCH_RESULT_ROOT:-${WPBENCH_ROOT}/results/raw_runs}"
read -r -a GPU_IDS <<< "${GPUS:-0}"

"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path rolling_forecast_config_report_dtw.json \
  --data-name-list 'forecasting_wp1_all_shapes_v1/tfb-Wind farm site 2 (Nominal capacity-200MW)_final.csv' \
  --strategy-args '{"strategy_name":"rolling_forecast","horizon":24,"tv_ratio":0.8,"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"stride":24,"num_rollings":48000,"seed":2021,"deterministic":"efficient","save_true_pred":false,"target_channel":[0]}' \
  --model-name tsfm.TinyTimeMixer \
  --adapter tinytimemixer_adapter \
  --model-hyper-params '{"few_shot_ratio":0.1,"finetune_style":"official_trainer","head_dropout":0.0,"horizon":48,"norm":true,"num_epochs":1,"official_batch_size":16,"official_learning_rate":0.0002110187415051424,"official_num_epochs":1,"official_patience":3,"patience":3,"pred_len":48,"prefix_horizon_slice":true,"seq_len":576,"shot_mode":"few_shot","trainer_num_workers":0,"trainer_seed":2021,"ttm_checkpoint_path":"checkpoints/tsfm/ttm_local/512-48-ft-r2.1","ttm_context_length":512}' \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS:-1}" \
  --num-cpus "${NUM_CPUS:-1}" \
  --gpus "${GPU_IDS[@]}" \
  --deterministic efficient \
  --timeout "${BENCHMARK_TIMEOUT:-60000}" \
  --save-path "${WPBENCH_RESULT_ROOT}/TinyTimeMixer/few_shot_10pct/tfb-Wind farm site 2 (Nominal capacity-200MW)_final/horizon_24"
