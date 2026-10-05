#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-gefcom2012.csv \
  --strategy-args '{"horizon":24,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":24,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name xlinear.Xlinear \
  --model-hyper-params '{"batch_size":32,"dropout":0.3,"horizon":24,"lr":0.00016291330626531667,"norm":true,"num_epochs":10,"num_workers":0,"patch_len":24,"patience":3,"pred_len":24,"seq_len":576}' \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 7 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_overnight_gpu7_rerun/Xlinear/tfb-gefcom2012/horizon_24/backfill
