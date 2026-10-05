# Reproduction Guide

All commands assume:

```bash
cd /home/wpbench_51/wpbench_artifacts
export WPBENCH_ROOT=$PWD
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}
export WPBENCH_RESULT_ROOT=$PWD/results/raw_runs
```

## External Inputs

The GitHub repository contains code, configs, scripts, manifests, and restore
instructions. Large binary artifacts are hosted externally.

Dataset bundle:

https://drive.google.com/drive/folders/1j8siMZ-SG6Q2yrfC0hPeAjG28fqImCib?usp=sharing

Checkpoint bundle:

https://drive.google.com/drive/folders/1sK-XpZl1J7DGdZW_xoZsrbFnLrgFZiCe?usp=sharing

After downloading the dataset bundle, restore it from the artifact root:

```bash
mkdir -p dataset/forecasting/forecasting_wp1_all_shapes_v1
cp -a wpbench_26_bundle_20260611/forecasting_wpbench_26/*.csv \
  dataset/forecasting/forecasting_wp1_all_shapes_v1/
mkdir -p dataset/metadata
cp -a wpbench_26_bundle_20260611/metadata/* dataset/metadata/
```

After downloading the checkpoint bundle, restore it from the artifact root:

```bash
cp -a wpbench_6algos_checkpoints_20260611/checkpoints ./
```

## Included Code

- Runtime benchmark code is under `ts_benchmark/`.
- Experiment scripts are under `scripts/run_experiments/icde_full_scripts/`.
- Environment exports are under `environment/`.

## Run A Script

Example:

```bash
bash scripts/run_experiments/icde_full_scripts/DLinear/hpo/tfb-Yalova_final_ready/horizon_12/dlinear.sh
```

The copied scripts have been patched to use:

- `WPBENCH_ROOT` for the artifact root.
- `PYTHON_BIN` for the Python executable.
- `WPBENCH_RESULT_ROOT` for generated run outputs.
- `WPBENCH_LOG_ROOT` for shell log directories when scripts define logs.

## Script Manifests

- `scripts/run_experiments/icde_full_scripts/icde_full_scripts_manifest.csv`: all 3589 scripts.
- `scripts/run_experiments/icde_full_scripts/final_selected_overlay_manifest.csv`: 41 final conflict-resolution scripts overlaid onto the full tree.
- `scripts/run_experiments/icde_full_scripts/path_patch_log.csv`: path rewrites applied for portability.

## Checkpoints

Large foundation/STFM checkpoints are not bundled in GitHub. Scripts refer to
relative checkpoint locations under:

```text
checkpoints/tsfm/
checkpoints/stfm/
```

Place or mount the required checkpoint directories there before running models
such as Toto, TinyTimeMixer, SEMPO, FactoST, or OpenCity_STFM. The Google Drive
checkpoint bundle above contains the minimal set for those checkpoint-dependent
algorithms.

## Outputs

Historical aggregated result tables are not included. New outputs should be
written under `results/raw_runs/` through `WPBENCH_RESULT_ROOT`.
