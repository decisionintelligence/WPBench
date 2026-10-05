#!/usr/bin/env bash
set -euo pipefail

# One fixed-parameter experiment; no hyperparameter search.
WPBENCH_ROOT="${WPBENCH_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)}"
export WPBENCH_ROOT
cd "${WPBENCH_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export WPBENCH_RESULT_ROOT="${WPBENCH_RESULT_ROOT:-${WPBENCH_ROOT}/results/raw_runs}"
read -r -a GPU_IDS <<< "${GPUS:-0}"

"${PYTHON_BIN}" scripts/run_benchmark.py \
  --config-path rolling_forecast_config_report_dtw.json \
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-Wind_Farm_B_filtered_wide_long_fixed__1T1V_reduced_1T_1V_windpower.csv \
  --strategy-args '{"strategy_name":"rolling_forecast","horizon":144,"tv_ratio":0.8,"train_ratio_in_tv":{"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS04.csv":0.75,"PEMS08.csv":0.75,"PEMS03.csv":0.75,"PEMS07.csv":0.75,"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"stride":144,"num_rollings":48000,"seed":2021,"deterministic":"efficient","save_true_pred":false,"target_channel":[0]}' \
  --model-name tsfm.Toto \
  --adapter toto_adapter \
  --model-hyper-params '{"seq_len":576,"batch_size":1,"checkpoint_path":"checkpoints/tsfm/Toto-Open-Base-1.0","finetune_style":"benchmark","horizon":144,"norm":false,"num_samples":32,"pred_len":144,"samples_per_batch":32,"seed":2021,"shot_mode":"zero_shot","use_kv_cache":true}' \
  --metrics '{"name":"mae"}' '{"name":"mse"}' '{"name":"rmse"}' '{"name":"nmae"}' '{"name":"nmse"}' '{"name":"nrmse"}' '{"name":"nmbe"}' '{"name":"maape"}' '{"name":"mape"}' '{"name":"smape"}' '{"name":"mase"}' '{"name":"rmsse"}' '{"name":"wape"}' '{"name":"msmape"}' '{"name":"edf"}' '{"name":"opencity_mae"}' '{"name":"opencity_mse"}' '{"name":"opencity_rmse"}' '{"name":"mae_norm"}' '{"name":"mse_norm"}' '{"name":"rmse_norm"}' '{"name":"mape_norm"}' '{"name":"smape_norm"}' '{"name":"mase_norm"}' '{"name":"wape_norm"}' '{"name":"msmape_norm"}' '{"name":"dtw"}' '{"name":"ndtw"}' '{"name":"nddtw"}' \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS:-1}" \
  --num-cpus "${NUM_CPUS:-1}" \
  --gpus "${GPU_IDS[@]}" \
  --deterministic efficient \
  --timeout "${BENCHMARK_TIMEOUT:-60000}" \
  --save-path "${WPBENCH_RESULT_ROOT}/Toto/zero_shot/tfb-Wind_Farm_B_filtered_wide_long_fixed__1T1V_reduced_1T_1V_windpower/horizon_144"
