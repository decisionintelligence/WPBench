#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list 'forecasting_wp1_all_shapes_v1/tfb-Wind farm site 3 (Nominal capacity-99MW)_final.csv' \
  --strategy-args '{"horizon":144,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":144,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name tsfm.TinyTimeMixer \
  --model-hyper-params '{"finetune_style":"official_trainer","head_dropout":0.0,"horizon":192,"norm":true,"num_epochs":1,"official_batch_size":16,"official_learning_rate":0.0002110187415051424,"official_num_epochs":1,"official_patience":3,"patience":3,"pred_len":192,"prefix_horizon_slice":true,"seq_len":576,"shot_mode":"full_shot","trainer_num_workers":0,"trainer_seed":2021,"ttm_checkpoint_path":"checkpoints/tsfm/ttm_local/512-192-r2","ttm_context_length":512}' \
  --adapter tinytimemixer_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 5 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path '${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_gpu5_next8h_small2_nomoirai/TinyTimeMixer/full_shot/tfb-Wind farm site 3 (Nominal capacity-99MW)_final/horizon_144/backfill'
