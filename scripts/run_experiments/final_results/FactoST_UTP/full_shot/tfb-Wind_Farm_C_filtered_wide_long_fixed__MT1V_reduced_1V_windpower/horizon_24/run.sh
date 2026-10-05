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
  --data-name-list forecasting_wp1_all_shapes_v1/tfb-Wind_Farm_C_filtered_wide_long_fixed__MT1V_reduced_1V_windpower.csv \
  --strategy-args '{"strategy_name":"rolling_forecast","horizon":24,"tv_ratio":0.8,"train_ratio_in_tv":{"AQShunyi.csv":0.75,"AQWan.csv":0.75,"ETTh1.csv":0.75,"ETTh1_spatial.csv":0.75,"ETTh2.csv":0.75,"ETTm1.csv":0.75,"ETTm2.csv":0.75,"PEMS03.csv":0.75,"PEMS04.csv":0.75,"PEMS07.csv":0.75,"PEMS08.csv":0.75,"Solar.csv":0.75,"__default__":0.875},"stride":24,"num_rollings":48000,"seed":2021,"deterministic":"efficient","save_true_pred":false,"target_channel":[0]}' \
  --model-name stfm.models.FactoST_UTP_Finetune \
  --adapter factost_utp_finetune_adapter \
  --model-hyper-params '{"batch_size":8,"factost_d_ff":1024,"factost_d_model":256,"factost_dropout":0.0,"factost_n_heads":4,"factost_n_layers":3,"factost_num_token":3,"factost_revin":true,"horizon":24,"loss":"MAE","lr":0.00016291330626531667,"lr_type":"OneCycle","norm":false,"patch_len":16,"pred_len":24,"pretrain_model_path":"checkpoints/stfm/factost_utp2_tiny_4utp.pt","seq_len":576,"shot_mode":"full_shot","stride":16}' \
  --eval-backend sequential \
  --num-workers "${NUM_WORKERS:-1}" \
  --num-cpus "${NUM_CPUS:-1}" \
  --gpus "${GPU_IDS[@]}" \
  --deterministic efficient \
  --timeout "${BENCHMARK_TIMEOUT:-60000}" \
  --save-path "${WPBENCH_RESULT_ROOT}/FactoST_UTP/full_shot/tfb-Wind_Farm_C_filtered_wide_long_fixed__MT1V_reduced_1V_windpower/horizon_24"
