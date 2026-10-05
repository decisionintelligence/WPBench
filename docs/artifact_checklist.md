# Artifact Checklist

- Runtime benchmark and model adapters: `ts_benchmark/`.
- Training/evaluation entry point: `scripts/run_benchmark.py`.
- Final-result scripts: `scripts/run_experiments/final_results/`, 2968 shell files.
- One task per shell script, uniquely identified by model, dataset, mode, and horizon.
- Main-table coverage: 1752 tasks, twice the 876 numeric cells in Tables III/IV.
- Foundation adaptation coverage: 608 few-shot and 608 zero-shot tasks.
- Task-to-source mapping: `scripts/run_experiments/final_results/final_results_manifest.csv`.
- Machine-readable coverage and cleanup audit: `docs/final_script_coverage_audit.json`.
- External dataset and checkpoint restoration instructions: `docs/reproduce.md`.
- Environment exports: `environment/`.

All 2968 scripts passed shell syntax, command-argument capture, and benchmark CLI
parser checks. Final task coverage has zero missing, extra, or duplicate tasks.

Datasets and checkpoint weights must be restored separately. These checks did
not execute training or compare newly generated numeric results with the paper.
