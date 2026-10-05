#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-maelstrom_turbine_1_univariate.csv \
  --strategy-args '{"horizon":72,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":72,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name tsfm.Toto \
  --model-hyper-params '{"batch_size":1,"checkpoint_path":"checkpoints/tsfm/Toto-Open-Base-1.0","dropout":0.0,"finetune_style":"official_trainer","horizon":72,"norm":false,"num_epochs":1,"num_samples":32,"num_train_samples":32,"pred_len":72,"samples_per_batch":32,"seed":2021,"seq_len":576,"shot_mode":"full_shot","toto_num_train_samples":32,"toto_official_lr":2.1101874150514252e-05,"toto_official_num_train_samples":32,"toto_official_train_batch_size":1,"use_kv_cache":true}' \
  --adapter toto_adapter \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 4 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_heatmap_fix3_server_day1_blocks_parallel_nomoirai/Toto/full_shot/tfb-maelstrom_turbine_1_univariate/horizon_72/backfill
