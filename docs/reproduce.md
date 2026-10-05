# Reproduction Guide

The final scripts run training/evaluation with selected parameters. Restore the
external inputs and install a compatible environment before launching runs.

## External Inputs

Dataset bundle:

https://drive.google.com/drive/folders/1j8siMZ-SG6Q2yrfC0hPeAjG28fqImCib?usp=sharing

Checkpoint bundle:

https://drive.google.com/drive/folders/1sK-XpZl1J7DGdZW_xoZsrbFnLrgFZiCe?usp=sharing

From the project root, restore the dataset files and metadata:

```bash
mkdir -p dataset/forecasting/forecasting_wp1_all_shapes_v1
cp -a /path/to/wpbench_26_bundle_20260611/forecasting_wpbench_26/*.csv \
  dataset/forecasting/forecasting_wp1_all_shapes_v1/
mkdir -p dataset/metadata
cp -a /path/to/wpbench_26_bundle_20260611/metadata/. dataset/metadata/
```

Restore checkpoint directories:

```bash
cp -a /path/to/wpbench_6algos_checkpoints_20260611/checkpoints/. checkpoints/
```

See `../checkpoints/README.md` for the checkpoint-dependent models and
`../environment/README.md` for environment notes. Large datasets and weights are
not committed to this repository.

## Run One Experiment

Activate the installed environment, then:

```bash
cd /path/to/WPBench_final
export PYTHON_BIN=python
export GPUS=0
bash scripts/run_experiments/final_results/DLinear/standard/tfb-Yalova_final_ready/horizon_12/run.sh
```

Scripts detect the project root automatically. `PYTHON_BIN` can point to a
specific interpreter. `GPUS` accepts space-separated GPU IDs; `NUM_WORKERS`,
`NUM_CPUS`, and `BENCHMARK_TIMEOUT` override resource settings.

Datasets are read from
`dataset/forecasting/forecasting_wp1_all_shapes_v1/`. To use a restored dataset
directory elsewhere, set `WPBENCH_FORECASTING_DATASET_PATH` to its parent
forecasting directory.

## Script Coverage

Scripts are under `scripts/run_experiments/final_results/` with the layout:

```text
<model>/<mode>/<dataset>/horizon_<H>/run.sh
```

Each script runs one task. The 1752 main-table tasks correspond to twice the
876 numeric Short/Long cells in Tables III and IV. Another 608 few-shot and
608 zero-shot tasks provide matching foundation adaptation comparisons for
Figure 11. Main-table full-shot tasks are shared by this comparison.

`final_results_manifest.csv` maps each task to its original source command and
records the selected parameters and evaluation strategy. The coverage audit is
in `docs/final_script_coverage_audit.json`. Duplicate sources were required to
agree on effective model parameters and evaluation strategy, and the 41 final
conflict overlays were preferred during selection.

## Outputs

By default, each task writes under:

```text
results/raw_runs/<model>/<mode>/<dataset>/horizon_<H>/
```

Set `WPBENCH_RESULT_ROOT` to change the output root. Reports use
`rolling_forecast_config_report_dtw.json`; it differs from the other bundled
config only in the reporting metric list. Explicit source metric lists are
preserved.

Historical result spreadsheets and plot-generation scripts are not bundled.
Shell syntax, command arguments, parser compatibility, and task coverage have
been checked; model training was not executed as part of script curation.
