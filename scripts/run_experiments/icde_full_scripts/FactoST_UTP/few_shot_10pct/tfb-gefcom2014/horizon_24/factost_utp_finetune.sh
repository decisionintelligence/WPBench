#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-gefcom2014.csv \
  --strategy-args '{"horizon":24,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":24,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --model-hyper-params '{"batch_size":8,"factost_d_ff":1024,"factost_d_model":256,"factost_dropout":0.0,"factost_n_heads":4,"factost_n_layers":3,"factost_num_token":3,"factost_revin":true,"few_shot_ratio":0.1,"horizon":24,"loss":"MAE","lr":0.0002110187415051424,"lr_type":"OneCycle","norm":false,"patch_len":16,"pred_len":24,"pretrain_model_path":"checkpoints/stfm/factost_utp2_tiny_4utp.pt","seq_len":576,"shot_mode":"few_shot","stride":16}' \
  --adapter factost_utp_finetune_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 4 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_day1_blocks_parallel_nomoirai/FactoST_UTP/few_shot_10pct/tfb-gefcom2014/horizon_24/backfill
