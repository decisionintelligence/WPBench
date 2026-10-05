#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-Wind_Farm_C_filtered_wide_long_fixed__MT1V_reduced_1V_windpower.csv \
  --strategy-args '{"horizon":144,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":144,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name stfm.models.FactoST_STA \
  --model-hyper-params '{"batch_size":8,"factost_dropout":0.0,"horizon":144,"lr":0.0002110187415051424,"norm":true,"pred_len":144,"seq_len":576,"shot_mode":"full_shot"}' \
  --adapter factost_sta_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 5 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_gpu5_next8h_small_nomoirai/FactoST_STA/full_shot/tfb-Wind_Farm_C_filtered_wide_long_fixed__MT1V_reduced_1V_windpower/horizon_144/backfill
