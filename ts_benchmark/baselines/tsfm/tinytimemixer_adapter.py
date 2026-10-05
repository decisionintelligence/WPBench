from typing import Type

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    TSFMDeepForecastingModelBase,
    generate_model_factory,
    TTM_HPARAMS,
)
from ts_benchmark.baselines.tsfm.tinytimemixer_helpers import (
    get_ttm_frequency_token,
    resolve_ttm_revision,
)
from ts_benchmark.baselines.tsfm.tinytimemixer_trainer_helpers import (
    prepare_ttm_trainer_datasets,
    run_ttm_official_trainer,
)


class TinyTimeMixerAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, TTM_HPARAMS, **kwargs)

        self.config.pred_len = int(
            getattr(self.config, "pred_len", getattr(self.config, "horizon", 96))
        )
        self.config.horizon = int(getattr(self.config, "horizon", self.config.pred_len))
        self.config.pred_len = self.config.horizon

        self.config.seq_len = int(getattr(self.config, "seq_len", 512))
        self.config.freq = str(getattr(self.config, "freq", "h"))
        self.config.target_dim = int(
            getattr(self.config, "target_dim", getattr(self.config, "c_out", 1))
        )

        if int(getattr(self.config, "label_len", 0)) <= 0:
            self.config.label_len = 0

        self.config.ttm_context_length = int(
            getattr(
                self.config,
                "ttm_context_length",
                getattr(self.config, "seq_len", 512),
            )
        )
        self.config.finetune_style = str(
            getattr(self.config, "finetune_style", "benchmark")
        ).lower()
        if self.config.finetune_style not in {"benchmark", "official_trainer"}:
            raise ValueError(
                f"Unknown finetune_style for TTM: {self.config.finetune_style}"
            )
        self.config.freeze_backbone_in_finetune = bool(
            getattr(self.config, "freeze_backbone_in_finetune", True)
        )
        official_lr = getattr(self.config, "official_learning_rate", None)
        self.config.official_learning_rate = (
            None if official_lr in {None, "", "None"} else float(official_lr)
        )
        self.config.official_batch_size = int(
            getattr(self.config, "official_batch_size", 64)
        )
        self.config.official_num_epochs = int(
            getattr(self.config, "official_num_epochs", 25)
        )
        self.config.official_patience = int(
            getattr(self.config, "official_patience", 10)
        )
        self.config.head_dropout = float(getattr(self.config, "head_dropout", 0.2))
        self.config.use_frequency_token = bool(
            getattr(self.config, "use_frequency_token", True)
        )
        self.config.use_official_lr_finder = bool(
            getattr(self.config, "use_official_lr_finder", False)
        )
        self.config.trainer_seed = int(getattr(self.config, "trainer_seed", 42))
        self.config.trainer_num_workers = int(
            getattr(self.config, "trainer_num_workers", 0)
        )
        self._ttm_freq_token_id = get_ttm_frequency_token(self.config.freq)

    def _on_shot_mode_resolved(self, mode: str) -> None:
        if mode in {"few_shot", "full_shot"}:
            self.config.lradj = "type1"

    def _apply_official_finetune_defaults(self, mode: str) -> None:
        if self.config.finetune_style != "official_trainer" or mode not in {
            "few_shot",
            "full_shot",
        }:
            return

        if int(getattr(self.config, "batch_size", 32)) == int(TTM_HPARAMS["batch_size"]):
            self.config.batch_size = self.config.official_batch_size
        if int(getattr(self.config, "num_epochs", 10)) == int(TTM_HPARAMS["num_epochs"]):
            self.config.num_epochs = self.config.official_num_epochs
        if int(getattr(self.config, "patience", 3)) == int(TTM_HPARAMS["patience"]):
            self.config.patience = self.config.official_patience
        if self.config.official_learning_rate is not None:
            self.config.lr = self.config.official_learning_rate

        self.config.freeze_backbone = self.config.freeze_backbone_in_finetune

    def _forecast_fit_with_official_trainer(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        train_dataset, valid_dataset = prepare_ttm_trainer_datasets(
            self,
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )
        self.model = self._init_model()
        self.CovariateFusion = None

        trained_native_model, learning_rate = run_ttm_official_trainer(
            self.model.model,
            train_dataset,
            valid_dataset,
            self.config,
        )
        self.config.lr = learning_rate
        self.model.model = trained_native_model
        self.check_point = self.save_checkpoint({"Model": self.model})
        return self

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        mode = self._resolve_shot_mode()
        self._apply_official_finetune_defaults(mode)

        _, native_pred_len, _ = resolve_ttm_revision(
            context_length=int(self.config.ttm_context_length),
            prediction_length=int(self.config.horizon),
            model_card=str(
                getattr(
                    self.config,
                    "ttm_model_card",
                    "ibm-granite/granite-timeseries-ttm-r2",
                )
            ),
            yaml_path=(str(getattr(self.config, "ttm_yaml_path", "")) or None),
        )

        if mode in {"few_shot", "full_shot"} and native_pred_len != int(self.config.horizon):
            raise ValueError(
                "Few/full-shot TTM requires model_hyper_params.horizon to match a native TTM checkpoint. "
                "If you want longer evaluation horizons, keep model_hyper_params.horizon at the native TTM length "
                "and set evaluation_config.strategy_args.horizon to the larger target horizon."
            )

        self._setup_post_norm_shot(mode)
        if self.config.finetune_style == "official_trainer" and mode in {"few_shot", "full_shot"}:
            return self._forecast_fit_with_official_trainer(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )

        return TSFMDeepForecastingModelBase.forecast_fit(
            self,
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )


def tinytimemixer_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=TinyTimeMixerAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )
