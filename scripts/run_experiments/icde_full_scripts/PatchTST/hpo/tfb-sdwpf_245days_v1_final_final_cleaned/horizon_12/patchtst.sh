#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config_report_dtw.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-sdwpf_245days_v1_final_final_cleaned.csv \
  --strategy-args '{"horizon":12,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":12,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name time_series_library.PatchTST \
  --model-hyper-params '{"batch_size":8,"d_ff":256,"dropout":0.2,"horizon":12,"lr":0.0022538779583357853,"norm":true,"patch_len":32,"pred_len":12,"seq_len":576}' \
  --adapter transformer_adapter \
  --metrics '{"name":"mae"}' '{"name":"mse"}' '{"name":"rmse"}' '{"name":"nmae"}' '{"name":"nmse"}' '{"name":"nrmse"}' '{"name":"nmbe"}' '{"name":"maape"}' '{"name":"mape"}' '{"name":"smape"}' '{"name":"mase"}' '{"name":"rmsse"}' '{"name":"wape"}' '{"name":"msmape"}' '{"name":"edf"}' '{"name":"opencity_mae"}' '{"name":"opencity_mse"}' '{"name":"opencity_rmse"}' '{"name":"mae_norm"}' '{"name":"mse_norm"}' '{"name":"rmse_norm"}' '{"name":"mape_norm"}' '{"name":"smape_norm"}' '{"name":"mase_norm"}' '{"name":"wape_norm"}' '{"name":"msmape_norm"}' '{"name":"dtw"}' '{"name":"ndtw"}' '{"name":"nddtw"}' \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 6 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/icde_missing_all_sheets_week_20260523/day29_patchtst_sdwpf_dff256_20260605/PatchTST/tfb-sdwpf_245days_v1_final_final_cleaned/horizon_12/backfill
