from typing import Any, Callable, Dict, List, Sequence


TrialLike = Any
SearchSpaceFn = Callable[[TrialLike], Dict[str, Any]]


def sample_lr(
    trial: TrialLike,
    low: float,
    high: float,
    *,
    name: str = "lr",
) -> float:
    return trial.suggest_float(name, low, high, log=True)


def sample_dropout(
    trial: TrialLike,
    name: str = "dropout",
    values: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
) -> float:
    return trial.suggest_categorical(name, list(values))


def sample_batch_size(
    trial: TrialLike,
    values: Sequence[int],
    *,
    name: str = "batch_size",
) -> int:
    return trial.suggest_categorical(name, list(values))


def sample_dlinear(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "moving_avg": trial.suggest_categorical("moving_avg", [13, 25, 49]),
    }


def sample_patchtst(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-2),
        "dropout": sample_dropout(trial),
        "patch_len": trial.suggest_categorical("patch_len", [8, 16, 24, 32]),
    }


def sample_itransformer(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-2),
        "dropout": sample_dropout(trial),
        "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
    }


def sample_duet(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-4, 3e-2),
        "dropout": sample_dropout(trial),
        "patch_len": trial.suggest_categorical("patch_len", [8, 16, 24, 32]),
    }


def sample_timebridge(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
        "attn_dropout": sample_dropout(trial, "attn_dropout"),
    }


def sample_xpatch(trial: TrialLike) -> Dict[str, Any]:
    patch_len = trial.suggest_categorical("patch_len", [8, 16, 24, 32])
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "patch_len": patch_len,
        "stride": patch_len // 2,
    }


def sample_amplifier(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "label_len": trial.suggest_categorical("label_len", [12, 24, 48]),
    }


def sample_crosslinear(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
        "patch_len": trial.suggest_categorical("patch_len", [8, 16, 24, 32]),
    }


def sample_xlinear(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
        "patch_len": trial.suggest_categorical("patch_len", [8, 16, 24, 32]),
    }


def sample_stwave(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
        "dims": trial.suggest_categorical("dims", [8, 16, 32]),
    }


def sample_stdn(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
        "K": trial.suggest_categorical("K", [8, 16, 32]),
    }


def sample_stssdl(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "dropout": sample_dropout(trial),
    }


def sample_tggc(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "rnn_units": trial.suggest_categorical("rnn_units", [32, 64, 128]),
        "cheb_order": trial.suggest_categorical("cheb_order", [2, 3, 4]),
    }


def sample_stid(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "node_dim": trial.suggest_categorical("node_dim", [16, 32, 64]),
        "embed_dim": trial.suggest_categorical("embed_dim", [16, 32, 64]),
    }


def sample_patchstg(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
    }


def sample_timer_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "batch_size": sample_batch_size(trial, [16, 32, 64]),
        "dropout": sample_dropout(trial),
    }


def sample_sempo_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "batch_size": sample_batch_size(trial, [32, 64, 128]),
    }


def sample_toto_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "toto_official_lr": sample_lr(trial, 1e-6, 1e-4, name="toto_official_lr"),
        "toto_official_train_batch_size": sample_batch_size(
            trial, [1, 2, 4], name="toto_official_train_batch_size"
        ),
        "dropout": sample_dropout(trial, values=(0.0, 0.05, 0.1, 0.2)),
    }


def sample_moirai_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-7, 5e-5),
        "batch_size": sample_batch_size(trial, [4, 8, 16]),
        "num_samples": trial.suggest_categorical("num_samples", [20, 50, 100]),
    }


def sample_ttm_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "official_learning_rate": sample_lr(
            trial, 1e-5, 1e-3, name="official_learning_rate"
        ),
        "official_batch_size": sample_batch_size(
            trial, [16, 32, 64], name="official_batch_size"
        ),
        "head_dropout": sample_dropout(trial, "head_dropout"),
    }


def sample_factost_sta_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "batch_size": sample_batch_size(trial, [8, 16, 32]),
        "factost_dropout": sample_dropout(trial, "factost_dropout"),
    }


def sample_factost_utp_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "batch_size": sample_batch_size(trial, [8, 16, 32]),
        "factost_dropout": sample_dropout(trial, "factost_dropout"),
    }


def sample_opencity_finetune(trial: TrialLike) -> Dict[str, Any]:
    return {
        "lr": sample_lr(trial, 1e-5, 1e-3),
        "batch_size": sample_batch_size(trial, [1, 2, 4]),
    }


SEARCH_SPACE_REGISTRY: Dict[str, SearchSpaceFn] = {
    "time_series_library.DLinear": sample_dlinear,
    "time_series_library.PatchTST": sample_patchtst,
    "time_series_library.iTransformer": sample_itransformer,
    "duet.DUET": sample_duet,
    "timebridge.TimeBridge": sample_timebridge,
    "xpatch.xPatch": sample_xpatch,
    "xlinear.Xlinear": sample_xlinear,
    "amplifier.Amplifier": sample_amplifier,
    "crosslinear.CrossLinear": sample_crosslinear,
    "st_model.STWave": sample_stwave,
    "st_model.STDN": sample_stdn,
    "st_model.STSSDL": sample_stssdl,
    "st_model.TGGC": sample_tggc,
    "st_model.STID": sample_stid,
    "st_model.PatchSTG": sample_patchstg,
    "tsfm.Timer": sample_timer_finetune,
    "tsfm.SEMPO": sample_sempo_finetune,
    "tsfm.Toto": sample_toto_finetune,
    "tsfm.Moirai": sample_moirai_finetune,
    "tsfm.TinyTimeMixer": sample_ttm_finetune,
    "stfm.models.FactoST_STA": sample_factost_sta_finetune,
    "stfm.models.FactoST_UTP": sample_factost_utp_finetune,
    "stfm.models.FactoST_UTP_Finetune": sample_factost_utp_finetune,
    "stfm.models.OpenCity_STFM": sample_opencity_finetune,
}


OFFICIAL_TUNING_REGISTRY: Dict[str, SearchSpaceFn] = {}


SEARCH_SPACE_METADATA: Dict[str, Dict[str, Any]] = {
    "time_series_library.DLinear": {
        "adapter": "transformer_adapter",
        "category": "regular_ts",
        "search_params": ["lr", "moving_avg"],
        "candidate_later": ["dropout", "d_model", "d_ff"],
    },
    "time_series_library.PatchTST": {
        "adapter": "transformer_adapter",
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "patch_len"],
        "candidate_later": ["d_model", "d_ff", "stride"],
    },
    "time_series_library.iTransformer": {
        "adapter": "transformer_adapter",
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "d_model"],
        "candidate_later": ["d_ff", "e_layers"],
    },
    "duet.DUET": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "patch_len"],
        "candidate_later": ["fc_dropout", "d_model", "d_ff"],
    },
    "timebridge.TimeBridge": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "attn_dropout"],
        "candidate_later": ["fc_dropout", "d_model", "d_ff"],
    },
    "xpatch.xPatch": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "patch_len", "stride"],
        "candidate_later": ["alpha", "beta", "ma_type"],
    },
    "xlinear.Xlinear": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "patch_len"],
        "candidate_later": ["d_model", "d_ff", "moving_avg"],
    },
    "amplifier.Amplifier": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "hidden_size", "label_len"],
        "candidate_later": ["num_epochs", "patience"],
    },
    "crosslinear.CrossLinear": {
        "adapter": None,
        "category": "regular_ts",
        "search_params": ["lr", "dropout", "patch_len"],
        "candidate_later": ["d_model", "d_ff", "moving_avg"],
    },
    "st_model.STWave": {
        "adapter": "stmodel_adapter",
        "category": "spatial",
        "search_params": ["lr", "dropout", "dims"],
        "candidate_later": ["heads", "level"],
    },
    "st_model.STDN": {
        "adapter": "stdn",
        "category": "spatial",
        "search_params": ["lr", "dropout", "K"],
        "candidate_later": ["L", "d", "order"],
    },
    "st_model.STSSDL": {
        "adapter": "stssdl",
        "category": "spatial",
        "search_params": ["lr", "dropout"],
        "candidate_later": ["adj_type"],
    },
    "st_model.TGGC": {
        "adapter": "stmodel_adapter",
        "category": "spatial",
        "search_params": ["lr", "rnn_units", "cheb_order"],
        "candidate_later": ["num_layers", "embed_dim"],
    },
    "st_model.STID": {
        "adapter": "stmodel_adapter",
        "category": "spatial",
        "search_params": ["lr", "node_dim", "embed_dim"],
        "candidate_later": ["temp_dim_tid", "temp_dim_diw"],
    },
    "st_model.PatchSTG": {
        "adapter": "patchstg_adapter",
        "category": "spatial",
        "search_params": ["lr"],
        "candidate_later": ["tem_patchsize", "dropout"],
    },
    "tsfm.Timer": {
        "adapter": "timer_adapter",
        "category": "foundation_ts",
        "search_params": ["lr", "batch_size", "dropout"],
        "candidate_later": ["weight_decay", "use_weight_decay"],
    },
    "tsfm.SEMPO": {
        "adapter": "sempo_adapter",
        "category": "foundation_ts",
        "search_params": ["lr", "batch_size"],
        "candidate_later": ["freeze_backbone", "domain_len", "d_model"],
    },
    "tsfm.Toto": {
        "adapter": "toto_adapter",
        "category": "foundation_ts",
        "search_params": ["toto_official_lr", "toto_official_train_batch_size", "dropout"],
        "candidate_later": ["toto_num_train_samples", "toto_loss_lambda_nll"],
    },
    "tsfm.Moirai": {
        "adapter": "moirai_adapter",
        "category": "foundation_ts",
        "search_params": ["lr", "batch_size", "num_samples"],
        "candidate_later": ["finetune_pattern", "weight_decay", "num_warmup_steps"],
    },
    "tsfm.TinyTimeMixer": {
        "adapter": "tinytimemixer_adapter",
        "category": "foundation_ts",
        "search_params": ["official_learning_rate", "official_batch_size", "head_dropout"],
        "candidate_later": ["freeze_backbone_in_finetune", "use_frequency_token"],
    },
    "stfm.models.FactoST_STA": {
        "adapter": "factost_sta_adapter",
        "category": "foundation_st",
        "search_params": ["lr", "batch_size", "factost_dropout"],
        "candidate_later": ["freeze_backbone", "factost_revin"],
    },
    "stfm.models.FactoST_UTP": {
        "adapter": "factost_utp_adapter",
        "category": "foundation_st",
        "search_params": ["lr", "batch_size", "factost_dropout"],
        "candidate_later": ["freeze_backbone", "factost_n_layers"],
    },
    "stfm.models.FactoST_UTP_Finetune": {
        "adapter": "factost_utp_finetune_adapter",
        "category": "foundation_st",
        "search_params": ["lr", "batch_size", "factost_dropout"],
        "candidate_later": ["freeze_backbone", "factost_n_layers"],
    },
    "stfm.models.OpenCity_STFM": {
        "adapter": "opencity_adapter",
        "category": "foundation_st",
        "search_params": ["lr", "batch_size"],
        "candidate_later": ["shot_mode", "few_shot_ratio"],
    },
}


MODEL_TUNING_TODO: List[Dict[str, str]] = [
    {
        "model": model_name.rsplit(".", 1)[-1],
        "model_name": model_name,
        "status": "supported",
    }
    for model_name in sorted(SEARCH_SPACE_REGISTRY)
]
MODEL_TUNING_TODO.extend(
    [
        {
            "model": "LR",
            "model_name": "time_series_library.LR",
            "status": "unsupported official tuning by scope",
        },
        {
            "model": "TimeXer",
            "model_name": "timexer.TimeXer",
            "status": "unsupported official tuning by scope",
        },
        {
            "model": "VAR",
            "model_name": "self_impl.VAR_model",
            "status": "unsupported official tuning by scope",
        },
    ]
)


def sample_params(model_name: str, trial: TrialLike, use_official_tuning: bool = False) -> Dict[str, Any]:
    registry = OFFICIAL_TUNING_REGISTRY if use_official_tuning else SEARCH_SPACE_REGISTRY
    if model_name not in registry:
        supported = ", ".join(sorted(registry))
        registry_name = "OFFICIAL_TUNING_REGISTRY" if use_official_tuning else "SEARCH_SPACE_REGISTRY"
        raise NotImplementedError(
            f"No HPO search space registered for {model_name!r} in {registry_name}. "
            f"Supported models: {supported or 'none'}. Add a JSON-serializable sampler "
            "function to ts_benchmark/hpo/search_space.py."
        )
    return registry[model_name](trial)


def has_official_tuning(model_name: str) -> bool:
    return model_name in OFFICIAL_TUNING_REGISTRY


def has_search_space(model_name: str) -> bool:
    return model_name in SEARCH_SPACE_REGISTRY
