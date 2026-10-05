# Environment

The requested source environment was `/opt/conda/envs/wpbench_unified_hpo`. This artifact includes:

- `wpbench_unified_hpo.yml`: conda YAML export with no build strings.
- `wpbench_unified_hpo_explicit.txt`: explicit conda package list.
- `wpbench_unified_hpo_pip_freeze.txt`: pip freeze.

Recommended local use in this checkout:

```bash
/opt/conda/envs/wpbench_unified_hpo/bin/python -V
```

Portable recreation, subject to CUDA/package availability:

```bash
conda env create -f environment/wpbench_unified_hpo.yml
```
