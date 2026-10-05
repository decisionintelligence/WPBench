import argparse
import json
import logging
import os
import subprocess
import sys
from typing import List, Optional


sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.hpo import run_optuna_search
from ts_benchmark.hpo.optuna_search import DEFAULT_WITH_DTW_METRICS


DEFAULT_FORECAST_LENGTHS = [12, 24, 72, 144]
DEFAULT_SEED = 2021


def _parse_forecast_lengths(value: Optional[str]) -> List[int]:
    if value is None or str(value).strip() == "":
        return list(DEFAULT_FORECAST_LENGTHS)
    parts = [part.strip() for part in str(value).replace(",", " ").split()]
    try:
        forecast_lengths = [int(part) for part in parts if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "forecast-lengths must be comma-separated integers, e.g. 12,24,72,144"
        ) from exc
    if not forecast_lengths:
        raise argparse.ArgumentTypeError("forecast-lengths must not be empty")
    return forecast_lengths


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def _build_auto_save_path(data_name_list: List[str], model_name: str) -> str:
    primary_series = data_name_list[0] if data_name_list else "unknown_series"
    primary_series = os.path.splitext(primary_series.replace(os.sep, "_"))[0]
    model_short_name = model_name.split(".")[-1] if model_name else "unknown_model"
    return os.path.join("hpo", primary_series, model_short_name)


def _run_backfill_script(backfill_script_path: Optional[str]) -> None:
    if not backfill_script_path:
        raise ValueError("backfill script path is missing from the HPO result")
    subprocess.run(["bash", backfill_script_path], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter optimization for the wind benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-path", required=True, help="Config JSON path.")
    parser.add_argument("--data-name-list", nargs="+", required=True, help="Series names.")
    parser.add_argument("--model-name", required=True, help="Benchmark model name.")
    parser.add_argument("--adapter", default=None, help="Optional benchmark adapter name.")
    parser.add_argument(
        "--model-hyper-params",
        default=None,
        help="Fixed model hyperparameter JSON merged before Optuna trial params.",
    )
    parser.add_argument(
        "--strategy-args",
        default=None,
        help="Fixed strategy argument JSON merged into the benchmark strategy args.",
    )
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for sampler and benchmark.")
    parser.add_argument(
        "--forecast-lengths",
        type=_parse_forecast_lengths,
        default=list(DEFAULT_FORECAST_LENGTHS),
        help="Comma-separated wind forecast lengths, e.g. 12,24,72,144.",
    )
    parser.add_argument("--save-path", default=None, help="Output path under result/.")
    parser.add_argument("--objective-metric", default="nmse", help="Metric minimized by Optuna.")
    parser.add_argument("--sampler", default="tpe", choices=["tpe", "random"], help="Sampler.")
    parser.add_argument("--timeout", type=float, default=None, help="Optuna timeout in seconds.")
    parser.add_argument("--eval-backend", choices=["sequential", "ray"], default="sequential")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of benchmark workers.")
    parser.add_argument("--num-cpus", type=int, default=1, help="Number of CPUs for benchmark backend.")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU ids for benchmark backend.")
    parser.add_argument(
        "--benchmark-timeout",
        type=float,
        default=60000,
        help="Timeout in seconds for each benchmark evaluation task.",
    )
    parser.add_argument("--study-name", default=None, help="Optuna study name.")
    parser.add_argument(
        "--use-official-tuning",
        type=_str_to_bool,
        default=False,
        help="Use official tuning registry when available.",
    )
    parser.add_argument(
        "--run-backfill-after-hpo",
        type=_str_to_bool,
        default=False,
        help="Execute the generated backfill benchmark script after HPO finishes.",
    )
    parser.add_argument(
        "--backfill-config-path",
        default=None,
        help="Optional config JSON used only by the generated best-params backfill script.",
    )
    parser.add_argument(
        "--backfill-include-dtw",
        type=_str_to_bool,
        default=False,
        help="Include dtw/ndtw/nddtw in the generated best-params backfill script.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    save_path = args.save_path or _build_auto_save_path(args.data_name_list, args.model_name)
    try:
        fixed_model_hyper_params = (
            json.loads(args.model_hyper_params) if args.model_hyper_params else None
        )
        fixed_strategy_args = json.loads(args.strategy_args) if args.strategy_args else None
    except json.JSONDecodeError as exc:
        parser.exit(status=2, message=f"invalid JSON argument: {exc}\n")

    try:
        result = run_optuna_search(
            config_path=args.config_path,
            data_name_list=args.data_name_list,
            model_name=args.model_name,
            adapter=args.adapter,
            fixed_model_hyper_params=fixed_model_hyper_params,
            fixed_strategy_args=fixed_strategy_args,
            save_path=save_path,
            n_trials=args.n_trials,
            seed=args.seed,
            forecast_lengths=args.forecast_lengths,
            objective_metric=args.objective_metric,
            sampler=args.sampler,
            timeout=args.timeout,
            eval_backend=args.eval_backend,
            num_workers=args.num_workers,
            num_cpus=args.num_cpus,
            gpus=args.gpus,
            benchmark_timeout=args.benchmark_timeout,
            study_name=args.study_name,
            use_official_tuning=args.use_official_tuning,
            backfill_config_path=args.backfill_config_path,
            backfill_metric_names=DEFAULT_WITH_DTW_METRICS if args.backfill_include_dtw else None,
        )
    except Exception as exc:
        parser.exit(status=2, message=f"run_hpo failed: {exc}\n")

    if args.run_backfill_after_hpo:
        backfill_script = result.get("backfill_script") or {}
        try:
            _run_backfill_script(backfill_script.get("path"))
        except Exception as exc:
            parser.exit(status=2, message=f"run_backfill failed: {exc}\n")
        backfill_script["executed"] = True
        result["backfill_script"] = backfill_script

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
