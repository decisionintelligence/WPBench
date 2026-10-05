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
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-gefcom2014.csv \
  --strategy-args '{"strategy_name":"rolling_forecast","horizon":144,"tv_ratio":0.8,"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"stride":144,"num_rollings":48000,"seed":2021,"deterministic":"efficient","save_true_pred":false,"target_channel":[0]}' \
  --model-name tsfm.Timer \
  --adapter timer_adapter \
  --model-hyper-params '{"batch_size":16,"dropout":0.0,"few_shot_ratio":0.1,"horizon":144,"lr":0.0002110187415051424,"norm":true,"pred_len":144,"seq_len":576,"shot_mode":"few_shot"}' \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS:-1}" \
  --num-cpus "${NUM_CPUS:-1}" \
  --gpus "${GPU_IDS[@]}" \
  --deterministic efficient \
  --timeout "${BENCHMARK_TIMEOUT:-60000}" \
  --save-path "${WPBENCH_RESULT_ROOT}/Timer/few_shot_10pct/tfb-gefcom2014/horizon_144"
