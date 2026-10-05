import copy
from datetime import datetime
import json
import logging
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import optuna
import torch

from ts_benchmark.common.constant import CONFIG_PATH, ROOT_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.recording import read_record_file
from ts_benchmark.utils.parallel import ParallelBackend

from ts_benchmark.hpo.search_space import has_official_tuning
from ts_benchmark.hpo.search_space import has_search_space
from ts_benchmark.hpo.search_space import sample_params
from ts_benchmark.evaluation.metrics import regression_metrics


WINDOWS_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
logger = logging.getLogger(__name__)

MODEL_HORIZON_OVERRIDE_KEY = "__hpo_model_horizon"
MODEL_PRED_LEN_OVERRIDE_KEY = "__hpo_model_pred_len"
BACKFILL_STRATEGY_EXCLUDE_KEYS = {"deterministic", "hpo_eval_mode"}
DTW_METRICS = ["dtw", "ndtw", "nddtw"]
DEFAULT_WITH_DTW_METRICS = list(regression_metrics.DEFAULT_METRICS) + DTW_METRICS


def _init_hpo_worker(env: dict) -> None:
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(3)


def _is_abs_path(path: str) -> bool:
    return os.path.isabs(path) or WINDOWS_ABS_PATH_RE.match(path) is not None


def _resolve_config_path(config_path: str) -> str:
    if _is_abs_path(config_path):
        return config_path
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    candidate = os.path.join(CONFIG_PATH, config_path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Config file not found: {config_path}")


def _resolve_output_dir(save_path: str) -> str:
    if not save_path:
        return os.path.join(ROOT_PATH, "result", "hpo")
    if _is_abs_path(save_path):
        return save_path
    normalized = os.path.normpath(save_path)
    if normalized == "result" or normalized.startswith("result" + os.sep):
        return os.path.join(ROOT_PATH, normalized)
    return os.path.join(ROOT_PATH, "result", normalized)


def _sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _model_script_name(model_name: str) -> str:
    model_short_name = str(model_name).split(".")[-1].lower()
    return f"{_sanitize_path_component(model_short_name)}.sh"


def _backfill_save_path(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "backfill")


def _config_name_for_run_benchmark(config_path: str) -> str:
    return os.path.basename(config_path)


def _format_cli_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _shell_command(parts: List[str]) -> str:
    if len(parts) <= 2:
        return " ".join(shlex.quote(str(part)) for part in parts)

    lines = [" ".join(shlex.quote(str(part)) for part in parts[:2])]
    index = 2
    while index < len(parts):
        option_parts = [parts[index]]
        index += 1
        while index < len(parts) and not str(parts[index]).startswith("--"):
            option_parts.append(parts[index])
            index += 1
        lines.append("  " + " ".join(shlex.quote(str(part)) for part in option_parts))
    return " \\\n".join(lines)


def _ordered_backfill_horizons(hpo_result: dict) -> List[int]:
    forecast_lengths = hpo_result.get("forecast_lengths")
    if forecast_lengths:
        return [int(item) for item in forecast_lengths]
    best_params = hpo_result.get("best_params_for_benchmark", {})
    return sorted(int(item) for item in best_params.keys())


def _write_backfill_script(
    output_dir: str,
    hpo_result: dict,
    config_path: str,
    config_data: dict,
    gpus: Optional[List[int]],
    num_workers: int,
    num_cpus: int,
    benchmark_timeout: float,
    python_executable: Optional[str] = None,
    eval_backend: str = "sequential",
    metric_names: Optional[List[str]] = None,
) -> str:
    model_name = hpo_result["model_name"]
    adapter = hpo_result.get("adapter")
    best_params_by_horizon = hpo_result.get("best_params_for_benchmark", {})
    if not best_params_by_horizon:
        raise ValueError("Cannot generate backfill script without best_params_for_benchmark.")

    base_strategy_args = copy.deepcopy(
        config_data.get("evaluation_config", {}).get("strategy_args", {}) or {}
    )
    deterministic = base_strategy_args.get("deterministic")
    target_channel = base_strategy_args.get("target_channel")
    script_path = os.path.join(output_dir, _model_script_name(model_name))
    backfill_save_path = _backfill_save_path(output_dir)
    python_executable = python_executable or sys.executable

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(ROOT_PATH)}",
        "",
    ]
    for horizon in _ordered_backfill_horizons(hpo_result):
        horizon_key = str(horizon)
        if horizon_key not in best_params_by_horizon:
            available = ", ".join(sorted(best_params_by_horizon)) or "none"
            raise ValueError(
                f"Horizon {horizon_key!r} not found in best_params_for_benchmark. "
                f"Available horizons: {available}"
            )

        strategy_args = {
            key: value
            for key, value in copy.deepcopy(base_strategy_args).items()
            if key not in BACKFILL_STRATEGY_EXCLUDE_KEYS
        }
        strategy_args["horizon"] = horizon
        strategy_args_json = json.dumps(
            strategy_args, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

        command_parts = [
            python_executable,
            "./scripts/run_benchmark.py",
            "--config-path",
            _config_name_for_run_benchmark(config_path),
            "--data-name-list",
            *hpo_result["data_name_list"],
            "--strategy-args",
            strategy_args_json,
            "--model-name",
            model_name,
            "--model-hyper-params",
            best_params_by_horizon[horizon_key],
        ]
        if adapter is not None:
            command_parts.extend(["--adapter", adapter])
        if metric_names:
            command_parts.append("--metrics")
            command_parts.extend(
                json.dumps({"name": metric_name}, separators=(",", ":"))
                for metric_name in metric_names
            )
        command_parts.extend(
            [
                "--eval-backend",
                eval_backend,
                "--num-workers",
                str(num_workers),
                "--num-cpus",
                str(num_cpus),
            ]
        )
        if gpus is not None:
            command_parts.extend(["--gpus", *[str(item) for item in gpus]])
        if deterministic is not None:
            command_parts.extend(["--deterministic", str(deterministic)])
        command_parts.extend(
            [
                "--timeout",
                _format_cli_value(benchmark_timeout),
                "--save-path",
                backfill_save_path,
            ]
        )
        lines.append(_shell_command(command_parts))
        lines.append("")

    with open(script_path, "w") as file_obj:
        file_obj.write("\n".join(lines).rstrip() + "\n")
    os.chmod(script_path, 0o755)
    return script_path


def _normalize_forecast_lengths(forecast_lengths: Optional[List[int]], config_data: dict) -> List[int]:
    if forecast_lengths is None:
        cfg_data = config_data.get("data_config", {})
        cfg_lengths = (
            cfg_data.get("forecast_lengths")
            or cfg_data.get("forecast_length")
            or cfg_data.get("prediction_length")
        )
        if cfg_lengths is None:
            forecast_lengths = [12, 24, 72, 144]
        elif isinstance(cfg_lengths, (list, tuple)):
            forecast_lengths = [int(item) for item in cfg_lengths]
        else:
            forecast_lengths = [int(cfg_lengths)]
    forecast_lengths = [int(item) for item in forecast_lengths]
    if not forecast_lengths:
        raise ValueError("forecast_lengths must be non-empty")
    return forecast_lengths


def _extract_objective_from_logs(log_files: List[str], objective_metric: str) -> float:
    if not log_files:
        raise ValueError("No log files returned by pipeline.")
    values = []
    errors = []
    for log_file in log_files:
        df = read_record_file(log_file)
        if "log_info" in df.columns:
            errors.extend(str(item) for item in df["log_info"].dropna().tolist() if str(item))
        if objective_metric not in df.columns:
            raise ValueError(
                f"{objective_metric!r} not found in {log_file}. Available columns: {df.columns.tolist()}"
            )
        metric_values = df[objective_metric].dropna().astype(float)
        values.extend(metric_values.tolist())
    if not values:
        error_preview = "\n".join(errors[:3])
        raise ValueError(
            f"No finite values found for objective metric {objective_metric!r}. "
            f"Pipeline errors: {error_preview}"
        )
    return float(sum(values) / len(values))


def _cleanup_log_files(log_files: List[str]) -> None:
    for file_path in log_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            logger.warning("Failed to remove temporary log file: %s", file_path)


def _get_default_model_params(config_data: dict, model_name: str, adapter: Optional[str]) -> dict:
    model_config = config_data.get("model_config", {})
    recommend_params = copy.deepcopy(model_config.get("recommend_model_hyper_params", {}) or {})
    default_params = {}
    if "input_chunk_length" in recommend_params:
        default_params["seq_len"] = recommend_params["input_chunk_length"]
    if "output_chunk_length" in recommend_params:
        default_params["horizon"] = recommend_params["output_chunk_length"]
        default_params["pred_len"] = recommend_params["output_chunk_length"]
    if "norm" in recommend_params:
        default_params["norm"] = recommend_params["norm"]

    for model in model_config.get("models", []):
        if model.get("model_name") != model_name:
            continue
        if adapter is not None and model.get("adapter") != adapter:
            continue
        default_params.update(copy.deepcopy(model.get("model_hyper_params") or {}))
        break
    return default_params


def _merge_params(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base or {})
    merged.update(copy.deepcopy(override or {}))
    return merged


def _apply_model_horizon(params: dict, horizon: int) -> dict:
    model_params = copy.deepcopy(params)
    model_horizon = model_params.pop(MODEL_HORIZON_OVERRIDE_KEY, None)
    model_pred_len = model_params.pop(MODEL_PRED_LEN_OVERRIDE_KEY, None)

    if model_horizon is None:
        model_horizon = horizon
    if model_pred_len is None:
        model_pred_len = model_horizon

    model_params["horizon"] = int(model_horizon)
    model_params["pred_len"] = int(model_pred_len)
    return model_params


def _build_model_config(
    base_model_config: dict,
    model_name: str,
    adapter: Optional[str],
    params: dict,
    horizon: int,
) -> dict:
    model_config = copy.deepcopy(base_model_config)
    model_params = _apply_model_horizon(params, horizon)
    model_config["models"] = [
        {
            "adapter": adapter,
            "model_name": model_name,
            "model_hyper_params": model_params,
        }
    ]
    return model_config


def evaluate_params(
    params: dict,
    config_data: dict,
    data_name_list: List[str],
    model_name: str,
    adapter: Optional[str],
    save_path: str,
    forecast_lengths: List[int],
    objective_metric: str,
    seed: Optional[int],
    eval_mode: str = "val",
    return_details: bool = False,
) -> Tuple[float, Dict[int, float]]:
    if eval_mode not in {"val", "test"}:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    base_data_config = copy.deepcopy(config_data["data_config"])
    base_model_config = copy.deepcopy(config_data["model_config"])
    base_evaluation_config = copy.deepcopy(config_data["evaluation_config"])
    base_strategy_args = copy.deepcopy(base_evaluation_config.get("strategy_args", {}))
    per_horizon_values: Dict[int, float] = {}

    for horizon in forecast_lengths:
        data_config = copy.deepcopy(base_data_config)
        data_config["data_name_list"] = data_name_list

        evaluation_config = copy.deepcopy(base_evaluation_config)
        strategy_args = copy.deepcopy(base_strategy_args)
        strategy_args["horizon"] = horizon
        strategy_args["hpo_eval_mode"] = eval_mode
        if seed is not None:
            strategy_args["seed"] = seed
        evaluation_config["strategy_args"] = strategy_args
        evaluation_config["save_path"] = os.path.join(save_path, f"{eval_mode}_horizon_{horizon}")

        model_config = _build_model_config(
            base_model_config=base_model_config,
            model_name=model_name,
            adapter=adapter,
            params=params,
            horizon=horizon,
        )

        log_files = pipeline(data_config, model_config, evaluation_config)
        try:
            objective = _extract_objective_from_logs(log_files, objective_metric)
        finally:
            _cleanup_log_files(log_files)
        per_horizon_values[horizon] = objective

    aggregate_value = sum(per_horizon_values.values()) / len(per_horizon_values)
    if return_details:
        return aggregate_value, per_horizon_values
    return aggregate_value, {}


def _build_sampler(name: str, seed: Optional[int]) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {name}")


def _make_json_safe(params: dict) -> dict:
    safe = {}
    for key, value in params.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Non-finite value for {key}: {value}")
        safe[key] = value
    json.dumps(safe)
    return safe


def run_optuna_search(
    config_path: str,
    data_name_list: List[str],
    model_name: str,
    adapter: Optional[str],
    save_path: str,
    fixed_model_hyper_params: Optional[dict] = None,
    fixed_strategy_args: Optional[dict] = None,
    n_trials: int = 10,
    seed: Optional[int] = 42,
    forecast_lengths: Optional[List[int]] = None,
    objective_metric: str = "nmse",
    sampler: str = "tpe",
    timeout: Optional[float] = None,
    eval_backend: str = "sequential",
    num_workers: int = 1,
    num_cpus: int = 1,
    gpus: Optional[List[int]] = None,
    benchmark_timeout: float = 60000,
    study_name: Optional[str] = None,
    use_official_tuning: bool = False,
    backfill_config_path: Optional[str] = None,
    backfill_metric_names: Optional[List[str]] = None,
) -> dict:
    if n_trials <= 0:
        raise ValueError("n_trials must be greater than 0.")

    resolved_config_path = _resolve_config_path(config_path)
    resolved_backfill_config_path = (
        _resolve_config_path(backfill_config_path) if backfill_config_path else resolved_config_path
    )
    with open(resolved_config_path, "r") as file_obj:
        config_data = json.load(file_obj)

    if fixed_strategy_args:
        strategy_args = config_data.setdefault("evaluation_config", {}).setdefault(
            "strategy_args", {}
        )
        strategy_args.update(copy.deepcopy(fixed_strategy_args))

    forecast_lengths = _normalize_forecast_lengths(forecast_lengths, config_data)
    output_dir = _resolve_output_dir(save_path)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    temp_eval_dir = tempfile.mkdtemp(prefix=f"hpo_eval_{run_id}_")
    study_name = study_name or f"hpo_{_sanitize_path_component(model_name)}_{run_id}"

    official_requested = bool(use_official_tuning)
    official_used = official_requested and has_official_tuning(model_name)
    if official_requested and not official_used:
        logger.warning(
            "Official tuning requested for %s, but no official tuning registry entry exists; "
            "falling back to SEARCH_SPACE_REGISTRY.",
            model_name,
        )
    if not official_used and not has_search_space(model_name):
        raise NotImplementedError(
            f"No registry search space is available for {model_name!r}. "
            "Add a sampler to SEARCH_SPACE_REGISTRY or use a model that is already supported."
        )

    baseline_params = _get_default_model_params(config_data, model_name, adapter)
    baseline_params = _merge_params(baseline_params, fixed_model_hyper_params or {})
    trial_failures = []
    optuna_sampler = _build_sampler(sampler, seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna_sampler,
        study_name=study_name,
    )

    torch.set_num_threads(3)

    ParallelBackend().init(
        backend=eval_backend,
        n_workers=num_workers,
        n_cpus=num_cpus,
        gpu_devices=gpus,
        default_timeout=benchmark_timeout,
        worker_initializers=[_init_hpo_worker],
    )
    try:
        baseline_value, baseline_per_horizon = evaluate_params(
            baseline_params,
            config_data,
            data_name_list,
            model_name,
            adapter,
            temp_eval_dir,
            forecast_lengths,
            objective_metric,
            seed,
            eval_mode="val",
            return_details=True,
        )

        def objective(trial: optuna.trial.Trial) -> float:
            try:
                trial_params = sample_params(model_name, trial, use_official_tuning=official_used)
                trial_params = _make_json_safe(trial_params)
                full_params = _merge_params(baseline_params, trial_params)
                value, per_horizon = evaluate_params(
                    full_params,
                    config_data,
                    data_name_list,
                    model_name,
                    adapter,
                    temp_eval_dir,
                    forecast_lengths,
                    objective_metric,
                    seed,
                    eval_mode="val",
                    return_details=True,
                )
                trial.set_user_attr("per_horizon", per_horizon)
                return value
            except Exception as exc:
                failure = {
                    "trial_number": trial.number,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                trial_failures.append(failure)
                trial.set_user_attr("failure", failure)
                logger.exception("Trial %s failed", trial.number)
                return float("inf")

        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    finally:
        ParallelBackend().close(force=True)
        shutil.rmtree(temp_eval_dir, ignore_errors=True)

    best_params = dict(study.best_params)
    full_best_params = _merge_params(baseline_params, best_params)
    best_value = float(study.best_value)
    best_value_rechecked = best_value
    best_per_horizon = dict(study.best_trial.user_attrs.get("per_horizon", {}))

    final_test_value = None
    final_test_per_horizon = {}

    best_params_for_benchmark = {}
    for horizon in forecast_lengths:
        params = _merge_params(baseline_params, best_params)
        params = _apply_model_horizon(params, horizon)
        best_params_for_benchmark[str(horizon)] = json.dumps(
            params, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    os.makedirs(output_dir, exist_ok=True)
    result = {
        "model_name": model_name,
        "adapter": adapter,
        "series_name": data_name_list[0] if data_name_list else None,
        "data_name_list": data_name_list,
        "objective": objective_metric,
        "n_trials": n_trials,
        "forecast_lengths": forecast_lengths,
        "baseline_params": baseline_params,
        "baseline_value": baseline_value,
        "baseline_per_horizon": baseline_per_horizon,
        "best_value": best_value,
        "best_value_rechecked": best_value_rechecked,
        "best_per_horizon": best_per_horizon,
        "final_test_value": final_test_value,
        "final_test_per_horizon": final_test_per_horizon,
        "best_params": best_params,
        "best_params_for_benchmark": best_params_for_benchmark,
        "study": {
            "sampler": sampler,
            "seed": seed,
            "study_name": study_name,
            "trial_failures": trial_failures,
            "eval_backend": eval_backend,
            "num_workers": num_workers,
            "num_cpus": num_cpus,
            "gpus": gpus,
            "benchmark_timeout": benchmark_timeout,
        },
        "official_tuning": {
            "requested": official_requested,
            "used": official_used,
            "source": model_name if official_used else None,
        },
    }

    backfill_script_path = _write_backfill_script(
        output_dir=output_dir,
        hpo_result=result,
        config_path=resolved_backfill_config_path,
        config_data=config_data,
        gpus=gpus,
        num_workers=num_workers,
        num_cpus=num_cpus,
        benchmark_timeout=benchmark_timeout,
        eval_backend=eval_backend,
        metric_names=backfill_metric_names,
    )
    result["backfill_script"] = {
        "path": backfill_script_path,
        "forecast_lengths": forecast_lengths,
        "config_path": resolved_backfill_config_path,
        "metrics": backfill_metric_names or [],
    }

    series_key = _sanitize_path_component("_".join(data_name_list[:3]) or "unknown_series")
    model_key = _sanitize_path_component(model_name)
    horizon_key = "-".join(str(item) for item in forecast_lengths)
    output_path = os.path.join(
        output_dir,
        f"{model_key}_{series_key}_h{horizon_key}_{run_id}_best_params.json",
    )
    with open(output_path, "w") as file_obj:
        json.dump(result, file_obj, indent=2, ensure_ascii=False)
    result["output_file"] = output_path
    return result
