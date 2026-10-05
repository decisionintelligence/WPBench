def run_optuna_search(*args, **kwargs):
    from ts_benchmark.hpo.optuna_search import run_optuna_search as _run_optuna_search

    return _run_optuna_search(*args, **kwargs)


__all__ = ["run_optuna_search"]
