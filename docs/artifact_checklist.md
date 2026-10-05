# Artifact Checklist

## Organized

- Benchmark runtime code: `ts_benchmark/`.
- Python run entry points: `scripts/run_benchmark.py`, `scripts/run_hpo.py`, `scripts/run_hpo_backfill.py`.
- Full selected ICDE script tree: `scripts/run_experiments/icde_full_scripts/` with 3589 scripts.
- Final conflict choices overlaid: `scripts/run_experiments/icde_full_scripts/final_selected_overlay_manifest.csv` with 41 scripts.
- Selected 26 datasets: `data/forecasting/forecasting_wp1_all_shapes_v1/`.
- Dataset manifest: `data/metadata/forecasting_wpbench_26_selected_files.csv`.
- Environment exports: `environment/`.
- Checkpoint placeholders: `checkpoints/tsfm/`, `checkpoints/stfm/`.

## Removed From This Runnable Artifact

- Empty placeholder namespace `wpbench/`.
- Historical aggregated result tables and copied raw metric outputs.
- Historical aggregation scripts and standalone evaluation helper scripts.
- Duplicate smoke/generated/curated run-script directories superseded by `icde_full_scripts/`.
- Script conflict work directories such as `_conflicts`, `_tools`, and old review packages.

## Needs Manual Confirmation

- Foundation/STFM checkpoints must be provided separately under `checkpoints/`.
- Some models may still require GPU-specific resource tuning before large batch runs.
- The artifact contains scripts and data for rerunning experiments, not precomputed paper tables.
