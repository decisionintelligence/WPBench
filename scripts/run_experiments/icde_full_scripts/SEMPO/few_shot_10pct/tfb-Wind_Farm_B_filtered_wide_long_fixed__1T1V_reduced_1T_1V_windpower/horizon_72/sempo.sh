#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-Wind_Farm_B_filtered_wide_long_fixed__1T1V_reduced_1T_1V_windpower.csv \
  --strategy-args '{"horizon":72,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":72,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name tsfm.SEMPO \
  --model-hyper-params '{"batch_size":32,"c_in":1,"d_layers":3,"d_model":256,"domain_len":128,"e_layers":3,"few_shot_ratio":0.1,"freeze_backbone":true,"head_type":"prediction","horizon":96,"label_len":48,"lr":0.00016291330626531667,"norm":true,"patch_len":64,"pred_len":96,"pretrain_model_path":"checkpoints/tsfm/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth","seq_len":512,"shot_mode":"few_shot","stride":64}' \
  --adapter sempo_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 4 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_gpu45_fill/SEMPO/few_shot_10pct/tfb-Wind_Farm_B_filtered_wide_long_fixed__1T1V_reduced_1T_1V_windpower/horizon_72/backfill
