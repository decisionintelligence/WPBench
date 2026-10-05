# ICDE Full Experiment Scripts

This directory contains the unified WPBench ICDE experiment script tree.

- Source tree: `/home/wpbench_51/wpbench_icde_script/scripts_unified_lys_ml_fixed_archive_20260610`
- Final conflict overlay source: `scripts/run_experiments/final_selected_scripts/`
- Shell scripts: 3589
- Final selected conflict scripts overlaid: 41

Directory layout:

```text
<model>/<mode>/<dataset>/horizon_<H>/<script>.sh
```

Manifest files:

- `icde_full_scripts_manifest.csv`: all shell scripts in this directory.
- `final_selected_overlay_manifest.csv`: the 41 final conflict-resolution scripts copied over this tree.

These scripts intentionally preserve the historical command bodies from the ICDE
runs. Machine-specific paths should be reviewed before running on a different
server.

Root-level scripts copied from the source tree: `bigst.sh`, `patchtst.sh`, `stdn.sh`, `timer.sh`.
