# Scripts

- `run_benchmark.py`: fixed-parameter training and evaluation entry point.
- `run_experiments/final_results/`: 2968 final-result shell scripts, one experiment per script.
- `download_data/`: optional local data materialization helpers.
- `preprocess/`: optional dataset cleaning and graph metadata helpers.

The shell scripts cover the main tables and the foundation adaptation comparison.
Their manifest records model, dataset, mode, horizon, and original source command.
Hyperparameter-search scripts, duplicate entry points, and repeated experiment
scripts have been removed. Historical table aggregation and plot-generation
helpers are not bundled.
