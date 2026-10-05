#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-gefcom2012.csv \
  --strategy-args '{"horizon":144,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":144,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name stfm.models.FactoST_STA \
  --model-hyper-params '{"batch_size":8,"factost_dropout":0.0,"few_shot_ratio":0.1,"horizon":144,"lr":0.0002110187415051424,"norm":true,"pred_len":144,"seq_len":576,"shot_mode":"few_shot"}' \
  --adapter factost_sta_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 7 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_overnight_gpu7_rerun/FactoST_STA/few_shot_10pct/tfb-gefcom2012/horizon_144/backfill
