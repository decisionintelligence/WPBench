# WPBench Runnable Artifact

This directory is a cleaned WPBench runnable artifact. It keeps the benchmark
runtime code, selected ICDE experiment scripts, environment exports, and
manifests for the external datasets/checkpoints needed by the current result
spreadsheet. Historical aggregated result tables and intermediate
script-conflict work products are intentionally removed.

Large binary assets are hosted outside GitHub:

- WPBench 26 dataset bundle: https://drive.google.com/drive/folders/1j8siMZ-SG6Q2yrfC0hPeAjG28fqImCib?usp=sharing
- Six-algorithm checkpoint bundle: https://drive.google.com/drive/folders/1sK-XpZl1J7DGdZW_xoZsrbFnLrgFZiCe?usp=sharing

## Layout

- `configs/`: dataset and experiment configs.
- `data/forecasting/`: placeholder and download instructions for selected benchmark CSV datasets.
- `dataset/metadata/forecasting_wpbench_26_selected_files.csv`: dataset file manifest when the external dataset bundle is restored locally.
- `scripts/run_experiments/icde_full_scripts/`: 3589 selected experiment shell scripts.
- `scripts/run_benchmark.py`, `scripts/run_hpo.py`: benchmark entry points used by the shell scripts.
- `ts_benchmark/`: runtime benchmark package, model adapters, metrics, HPO, data loading, and reporting code.
- `checkpoints/`: placeholder for external TSFM/STFM checkpoints.
- `environment/`: exports from `/opt/conda/envs/wpbench_unified_hpo`.
- `results/raw_runs/`: empty output location for newly generated runs.
- `docs/`: reproduction notes and cleanup record.

## Quick Run

```bash
cd /home/wpbench_51/wpbench_artifacts
export WPBENCH_ROOT=$PWD
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}
export WPBENCH_RESULT_ROOT=$PWD/results/raw_runs
bash scripts/run_experiments/icde_full_scripts/DLinear/hpo/tfb-Yalova_final_ready/horizon_12/dlinear.sh
```

For full notes, see `docs/reproduce.md`.
