# Run Experiments

`final_results/` contains 2968 fixed-parameter experiment scripts in the layout:

```text
<model>/<mode>/<dataset>/horizon_<H>/run.sh
```

Each script runs exactly one horizon. The modes are:

- `standard`: 1144 ordinary training/evaluation tasks.
- `full_shot`: 608 foundation-model tasks used in the main tables and adaptation comparison.
- `few_shot_10pct`: 608 foundation adaptation tasks.
- `zero_shot`: 608 matching foundation reference tasks.

See `final_results/final_results_manifest.csv` for every retained task and its
source. The 41 final conflict-overlay scripts were preferred when selecting
sources. Duplicate candidates were checked for equal effective model parameters
and evaluation strategy. Paper-scope coverage is recorded in
`../../docs/final_script_coverage_audit.json`.

Recommended environment before running scripts:

```bash
cd /path/to/WPBench_final
export PYTHON_BIN=python
export GPUS=0
```

Project-root detection and the default output directory are handled by each
script. GPU IDs, worker count, CPU count, and timeout can be overridden through
`GPUS`, `NUM_WORKERS`, `NUM_CPUS`, and `BENCHMARK_TIMEOUT`.
