import argparse
import json
import os
import sys
from typing import List, Optional


sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from scripts.run_benchmark import build_data_config
from scripts.run_benchmark import build_evaluation_config
from scripts.run_benchmark import build_model_config
from scripts.run_benchmark import build_report_config
from scripts.run_benchmark import init_worker
from ts_benchmark.common.constant import CONFIG_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.report import report
from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.utils.parallel import ParallelBackend


def _str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _load_hpo_result(path: str) -> dict:
    with open(path, "r") as file_obj:
        return json.load(file_obj)


def _config_name_for_run_benchmark(config_path: str) -> str:
    return os.path.basename(config_path)


def _default_save_path(hpo_result: dict, horizon: int) -> str:
    model_name = str(hpo_result["model_name"]).split(".")[-1]
    return f"hpo_backfill/{model_name}/h{horizon}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill a run_benchmark evaluation from an HPO result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hpo-result", required=True, help="Path to run_hpo output JSON.")
    parser.add_argument("--horizon", type=int, required=True, help="Horizon key to backfill.")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Benchmark config filename/path. Defaults to basename of HPO config-path style.",
    )
    parser.add_argument("--save-path", default=None, help="Result save path under result/.")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ids for benchmark backend.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of benchmark workers.")
    parser.add_argument("--num-cpus", type=int, default=1, help="Number of CPUs for benchmark backend.")
    parser.add_argument("--timeout", type=float, default=60000, help="Benchmark timeout in seconds.")
    parser.add_argument("--eval-backend", choices=["sequential", "ray"], default="sequential")
    parser.add_argument("--deterministic", choices=["full", "efficient", "none"], default="efficient")
    parser.add_argument("--seed", type=int, default=None, help="Optional benchmark seed override.")
    parser.add_argument("--save-true-pred", type=_str_to_bool, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    hpo_result = _load_hpo_result(args.hpo_result)
    horizon_key = str(args.horizon)
    best_params_by_horizon = hpo_result.get("best_params_for_benchmark", {})
    if horizon_key not in best_params_by_horizon:
        available = ", ".join(sorted(best_params_by_horizon)) or "none"
        raise ValueError(f"Horizon {horizon_key!r} not found. Available horizons: {available}")

    config_path = args.config_path or _config_name_for_run_benchmark(
        hpo_result.get("config_path", "rolling_forecast_config.json")
    )
    with open(os.path.join(CONFIG_PATH, config_path), "r") as file_obj:
        config_data = json.load(file_obj)

    benchmark_args = argparse.Namespace(
        config_path=config_path,
        data_name_list=hpo_result["data_name_list"],
        data_set_name=None,
        adapter=[hpo_result.get("adapter")],
        model_name=[hpo_result["model_name"]],
        model_hyper_params=[best_params_by_horizon[horizon_key]],
        metrics=None,
        strategy_args=json.dumps({"horizon": args.horizon}),
        seed=args.seed,
        deterministic=args.deterministic,
        eval_backend=args.eval_backend,
        num_cpus=args.num_cpus,
        gpus=args.gpus,
        num_workers=args.num_workers,
        timeout=args.timeout,
        max_tasks_per_child=100,
        aggregate_type="mean",
        report_method="csv",
        save_path=args.save_path or _default_save_path(hpo_result, args.horizon),
        save_true_pred=args.save_true_pred,
    )

    data_config = build_data_config(benchmark_args, config_data)
    model_config = build_model_config(benchmark_args, config_data)
    evaluation_config = build_evaluation_config(benchmark_args, config_data)
    report_config = build_report_config(benchmark_args, config_data)

    ParallelBackend().init(
        backend=benchmark_args.eval_backend,
        n_workers=benchmark_args.num_workers,
        n_cpus=benchmark_args.num_cpus,
        gpu_devices=benchmark_args.gpus,
        default_timeout=benchmark_args.timeout,
        max_tasks_per_child=benchmark_args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )
    try:
        log_filenames = pipeline(data_config, model_config, evaluation_config)
    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    report_config["leaderboard_file_name"] = "test_report" + get_unique_file_suffix()
    report(report_config, report_method=benchmark_args.report_method)

    print(
        json.dumps(
            {
                "hpo_result": args.hpo_result,
                "horizon": args.horizon,
                "model_name": hpo_result["model_name"],
                "adapter": hpo_result.get("adapter"),
                "model_hyper_params": best_params_by_horizon[horizon_key],
                "save_path": benchmark_args.save_path,
                "log_files": log_filenames,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
