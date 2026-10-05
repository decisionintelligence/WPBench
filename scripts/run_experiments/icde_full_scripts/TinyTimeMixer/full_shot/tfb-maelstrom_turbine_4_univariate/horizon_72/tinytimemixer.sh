#!/usr/bin/env bash
set -euo pipefail
cd "${WPBENCH_ROOT:-/home/wpbench_51/wpbench_artifacts}"

"${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}" ./scripts/run_benchmark.py \
  --config-path rolling_forecast_config_report_dtw.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-maelstrom_turbine_4_univariate.csv \
  --strategy-args '{"horizon":72,"num_rollings":48000,"save_true_pred":false,"seed":2021,"strategy_name":"rolling_forecast","stride":72,"target_channel":[0],"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"tv_ratio":0.8}' \
  --model-name tsfm.TinyTimeMixer \
  --model-hyper-params '{"finetune_style":"official_trainer","head_dropout":0.0,"horizon":96,"norm":true,"num_epochs":1,"official_batch_size":16,"official_learning_rate":0.00016291330626531667,"official_num_epochs":1,"official_patience":3,"patience":3,"pred_len":96,"prefix_horizon_slice":true,"seq_len":576,"shot_mode":"full_shot","trainer_num_workers":0,"trainer_seed":2021,"ttm_checkpoint_path":"checkpoints/tsfm/ttm_local/512-96-ft-r2.1","ttm_context_length":512}' \
  --adapter tinytimemixer_adapter \
  --metrics '{"name":"mae"}' '{"name":"mse"}' '{"name":"rmse"}' '{"name":"nmae"}' '{"name":"nmse"}' '{"name":"nrmse"}' '{"name":"nmbe"}' '{"name":"maape"}' '{"name":"mape"}' '{"name":"smape"}' '{"name":"mase"}' '{"name":"rmsse"}' '{"name":"wape"}' '{"name":"msmape"}' '{"name":"edf"}' '{"name":"opencity_mae"}' '{"name":"opencity_mse"}' '{"name":"opencity_rmse"}' '{"name":"mae_norm"}' '{"name":"mse_norm"}' '{"name":"rmse_norm"}' '{"name":"mape_norm"}' '{"name":"smape_norm"}' '{"name":"mase_norm"}' '{"name":"wape_norm"}' '{"name":"msmape_norm"}' '{"name":"dtw"}' '{"name":"ndtw"}' '{"name":"nddtw"}' \
  --eval-backend sequential \
  --num-workers 1 \
  --num-cpus 1 \
  --gpus 4 \
  --deterministic efficient \
  --timeout 60000 \
  --save-path ${WPBENCH_RESULT_ROOT:-/home/wpbench_51/wpbench_artifacts/results/raw_runs}/hpo_green_fix4_remote_fast_4gpu_20260513/TinyTimeMixer/full_shot/tfb-maelstrom_turbine_4_univariate/horizon_72/backfill
