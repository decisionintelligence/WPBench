# Run Experiments

`icde_full_scripts/` is the authoritative experiment script tree for this artifact.
It contains 3589 shell scripts in the layout:

```text
<model>/<mode>/<dataset>/horizon_<H>/<script>.sh
```

The tree was built from the unified ICDE script archive and then overlaid with the
41 final selected conflict-resolution scripts. See:

- `icde_full_scripts/icde_full_scripts_manifest.csv`
- `icde_full_scripts/final_selected_overlay_manifest.csv`
- `icde_full_scripts/path_patch_log.csv`

Recommended environment before running scripts:

```bash
cd /home/wpbench_51/wpbench_artifacts
export WPBENCH_ROOT=$PWD
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/wpbench_unified_hpo/bin/python}
export WPBENCH_RESULT_ROOT=$PWD/results/raw_runs
```
