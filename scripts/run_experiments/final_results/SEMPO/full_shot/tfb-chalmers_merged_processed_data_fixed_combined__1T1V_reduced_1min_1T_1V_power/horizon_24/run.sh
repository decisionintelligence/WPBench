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
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-chalmers_merged_processed_data_fixed_combined__1T1V_reduced_1min_1T_1V_power.csv \
  --strategy-args '{"strategy_name":"rolling_forecast","horizon":24,"tv_ratio":0.8,"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"stride":24,"num_rollings":48000,"seed":2021,"deterministic":"efficient","save_true_pred":false,"target_channel":[0]}' \
  --model-name tsfm.SEMPO \
  --adapter sempo_adapter \
  --model-hyper-params '{"batch_size":32,"c_in":1,"d_layers":3,"d_model":256,"domain_len":128,"e_layers":3,"freeze_backbone":true,"head_type":"prediction","horizon":96,"label_len":48,"lr":0.00016291330626531667,"norm":true,"patch_len":64,"pred_len":96,"pretrain_model_path":"checkpoints/tsfm/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth","seq_len":512,"shot_mode":"full_shot","stride":64}' \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS:-1}" \
  --num-cpus "${NUM_CPUS:-1}" \
  --gpus "${GPU_IDS[@]}" \
  --deterministic efficient \
  --timeout "${BENCHMARK_TIMEOUT:-60000}" \
  --save-path "${WPBENCH_RESULT_ROOT}/SEMPO/full_shot/tfb-chalmers_merged_processed_data_fixed_combined__1T1V_reduced_1min_1T_1V_power/horizon_24"
