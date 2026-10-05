# Final-Result Experiments

This directory contains 2968 shell scripts. Each script runs one fixed-parameter
model/dataset/mode/horizon task through `scripts/run_benchmark.py`.

| Purpose | Scripts |
| --- | ---: |
| Table III main results | 932 |
| Table IV main results | 820 |
| Figure 11 few-shot adaptation | 608 |
| Figure 11 zero-shot reference | 608 |

The main results include 1144 standard tasks and 608 full-shot tasks. Full-shot
tasks are reused by the adaptation comparison. Horizons 12/24 form Short cells,
and horizons 72/144 form Long cells. Unavailable cells and repeated metrics do
not add tasks.

`final_results_manifest.csv` provides one row per script, including its paper
scope, original source, model parameters, evaluation strategy, and file hashes.
Source paths refer to the import baseline commit recorded in
`../../../docs/final_script_coverage_audit.json` and remain recoverable from Git.

Scripts use the selected parameters and the original evaluation strategy.
They detect the project root, use the documented dataset bundle location, and
write to a distinct output directory per task. `PYTHON_BIN` defaults to `python`,
and `GPUS` defaults to `0`. Other resource defaults are taken from the source
command and can be overridden through environment variables.

All scripts passed shell syntax, actual command-argument capture, and benchmark
CLI parser checks. Their task set matches the paper-scope manifest with no
missing, extra, or duplicate tasks. These checks do not execute model training.
