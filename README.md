# WPBench

WPBench is a wind-power forecasting benchmark covering 26 datasets and 19
models. This release includes fixed-parameter experiment scripts for the paper's
main results and foundation-model adaptation comparison. Trainable models are
trained and evaluated with the selected parameters; hyperparameter search is
not required.

Large binary assets are hosted outside GitHub:

- WPBench 26 dataset bundle: https://drive.google.com/drive/folders/1j8siMZ-SG6Q2yrfC0hPeAjG28fqImCib?usp=sharing
- Six-algorithm checkpoint bundle: https://drive.google.com/drive/folders/1sK-XpZl1J7DGdZW_xoZsrbFnLrgFZiCe?usp=sharing

## Layout

- `configs/`: experiment and reporting configs.
- `dataset/forecasting/forecasting_wp1_all_shapes_v1/`: restore the external dataset CSV files here.
- `scripts/run_experiments/final_results/`: 2968 scripts, each running one experiment.
- `scripts/run_benchmark.py`: training and evaluation entry point.
- `ts_benchmark/`: model adapters, metrics, data loading, evaluation, and reporting.
- `checkpoints/`: placeholder for external TSFM/STFM checkpoints.
- `environment/`: environment exports and installation notes.
- `results/raw_runs/`: generated experiment outputs.
- `docs/`: reproduction instructions and script coverage audit.

## Experiment Coverage

| Scope | Reported numeric cells | Individual experiment scripts |
| --- | ---: | ---: |
| Table III: single-turbine main results | 466 | 932 |
| Table IV: multi-turbine main results | 410 | 820 |
| Figure 11: few-shot adaptation | — | 608 |
| Figure 11: zero-shot reference | — | 608 |
| Total | | 2968 |

Each Short cell averages horizons 12 and 24; each Long cell averages horizons
72 and 144. Unavailable cells are excluded. Main-table foundation results use
full-shot adaptation; the same full-shot runs also support Figure 11 and are
counted once. Multiple metrics from one run do not require separate scripts.

The [final script manifest](scripts/run_experiments/final_results/final_results_manifest.csv)
maps every task to its script, fixed parameters, paper scope, and original
source command. The [coverage audit](docs/final_script_coverage_audit.json)
records the exclusions and verification results. Historical result tables and
plot-generation scripts are not bundled.

## Quick Run

```bash
cd /path/to/WPBench_final
export PYTHON_BIN=python
export GPUS=0
bash scripts/run_experiments/final_results/DLinear/standard/tfb-Yalova_final_ready/horizon_12/run.sh
```

Scripts detect the project root automatically. Set `WPBENCH_RESULT_ROOT` to
override the output location. Restore datasets and required checkpoints before
running; see [the reproduction guide](docs/reproduce.md).
