#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config_report_dtw.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-kaggle1_final_benchmark_v3_ready_no_turbinestatus.csv \
  --strategy-args '{"horizon":72,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":72,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name tsfm.SEMPO \
  --model-hyper-params '{"batch_size":32,"c_in":1,"d_layers":3,"d_model":256,"domain_len":128,"e_layers":3,"freeze_backbone":true,"head_type":"prediction","horizon":96,"label_len":48,"lr":0.00016291330626531667,"norm":true,"patch_len":64,"pred_len":96,"pretrain_model_path":"checkpoints/tsfm/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth","seq_len":512,"shot_mode":"full_shot","stride":64}' \
  --adapter sempo_adapter \
  --metrics '{"name":"mae"}' '{"name":"mse"}' '{"name":"rmse"}' '{"name":"nmae"}' '{"name":"nmse"}' '{"name":"nrmse"}' '{"name":"nmbe"}' '{"name":"maape"}' '{"name":"mape"}' '{"name":"smape"}' '{"name":"mase"}' '{"name":"rmsse"}' '{"name":"wape"}' '{"name":"msmape"}' '{"name":"edf"}' '{"name":"opencity_mae"}' '{"name":"opencity_mse"}' '{"name":"opencity_rmse"}' '{"name":"mae_norm"}' '{"name":"mse_norm"}' '{"name":"rmse_norm"}' '{"name":"mape_norm"}' '{"name":"smape_norm"}' '{"name":"mase_norm"}' '{"name":"wape_norm"}' '{"name":"msmape_norm"}' '{"name":"dtw"}' '{"name":"ndtw"}' '{"name":"nddtw"}' \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 6 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_fix4_remote_kaggle1_no_turbinestatus_gpu6_20260513/SEMPO/full_shot/tfb-kaggle1_final_benchmark_v3_ready_no_turbinestatus/horizon_72/backfill
