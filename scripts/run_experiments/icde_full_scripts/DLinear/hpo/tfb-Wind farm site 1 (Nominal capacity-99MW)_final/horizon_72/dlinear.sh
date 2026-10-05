#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list 'forecasting_wp1_all_shapes_v1/tfb-Wind farm site 1 (Nominal capacity-99MW)_final.csv' \
  --strategy-args '{"horizon":72,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":72,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name time_series_library.DLinear \
  --model-hyper-params '{"horizon":72,"lr":0.0006575591666965022,"moving_avg":13,"norm":true,"pred_len":72,"seq_len":576}' \
  --adapter transformer_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 4 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path '${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_gpu4_next8h_small2_nomoirai/DLinear/tfb-Wind farm site 1 (Nominal capacity-99MW)_final/horizon_72/backfill'
