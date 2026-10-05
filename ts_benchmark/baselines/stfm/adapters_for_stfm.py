from typing import Dict, Optional, Type

import pandas as pd
import torch
import torch.nn as nn

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.deep_forecasting_model_base import (
    DeepForecastingModelBase,
)
from ts_benchmark.baselines.stfm.few_shot_utils import apply_few_shot_loader

GENERIC_STFM_HPARAMS = {
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 10,
    "patience": 3,
    "label_len": 0,
    "norm": True,
    "shot_mode": "full_shot",
    "few_shot_ratio": 0.1,
    "sampling_basis": "sample",
    "sampling_strategy": "uniform",
    "freeze_backbone": False,
    "pretrain_model_path": "",
    "ckpt_path": "",
}

class _ZeroLoss(nn.Module):
    def forward(self, pred, target):
        del target
        return pred.new_zeros(())


class STFMAdapterWorkflowBase:
    def _on_shot_mode_resolved(self, mode: str) -> None:
        del mode

    def _setup_post_norm_shot(self, mode: str) -> None:
        if mode == "few_shot":
            self._sampling_rate()
            self._sampling_strategy()
            self.config.sampling_basis = "sample"
            self._sampling_basis()
            return
        if mode in {"zero_shot", "full_shot"}:
            return
        raise ValueError(f"Unknown shot_mode: {mode}")

    def _resolve_shot_mode(self) -> str:
        mode = str(getattr(self.config, "shot_mode", "full_shot")).lower()
        if mode not in {"few_shot", "zero_shot", "full_shot"}:
            raise ValueError(f"Unknown shot_mode: {mode}")

        self.config.shot_mode = mode
        self._on_shot_mode_resolved(mode)

        if mode == "few_shot":
            self.config.freeze_backbone = False
            if int(getattr(self.config, "num_epochs", 0)) <= 0:
                self.config.num_epochs = 2
        elif mode == "zero_shot":
            self.config.num_epochs = 0
            self.config.freeze_backbone = True
        else:
            self.config.freeze_backbone = False

        return mode

    def _sampling_rate(self) -> float:
        raw = float(
            getattr(
                self.config,
                "sampling_rate",
                getattr(self.config, "few_shot_ratio", 0.1),
            )
        )
        rate = max(0.0, min(1.0, raw))
        self.config.sampling_rate = rate
        self.config.few_shot_ratio = rate
        return rate

    def _sampling_strategy(self) -> str:
        strategy = str(getattr(self.config, "sampling_strategy", "uniform")).lower()
        if strategy not in {"begin", "end", "uniform", "random"}:
            raise ValueError(f"Unknown sampling_strategy: {strategy}")
        self.config.sampling_strategy = strategy
        return strategy

    def _sampling_basis(self) -> str:
        basis = str(getattr(self.config, "sampling_basis", "data")).lower()
        if basis not in {"data", "sample"}:
            raise ValueError(f"Unknown sampling_basis: {basis}")
        self.config.sampling_basis = basis
        return basis


class STFMDeepForecastingModelBase(DeepForecastingModelBase):
    @staticmethod
    def _without_exog_covariates(covariates: Optional[dict]) -> Optional[dict]:
        if not covariates:
            return covariates
        covariates = dict(covariates)
        covariates.pop("exog", None)
        return covariates or None

    class _TargetOnlyBatchMaker:
        def __init__(self, batch_maker):
            self._batch_maker = batch_maker

        def make_batch(self, *args, **kwargs):
            batch = self._batch_maker.make_batch(*args, **kwargs)
            if not isinstance(batch, dict):
                return batch
            covariates = batch.get("covariates")
            if not covariates or "exog" not in covariates:
                return batch
            batch = dict(batch)
            covariates = dict(covariates)
            covariates.pop("exog", None)
            batch["covariates"] = covariates or None
            return batch

        def __getattr__(self, name):
            return getattr(self._batch_maker, name)

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        covariates = self._without_exog_covariates(covariates)
        use_dataset_sample = (
            str(getattr(self.config, "shot_mode", "full_shot")).lower() == "few_shot"
        )
        if not use_dataset_sample:
            return super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )

        original_provider = deep_base_module.forecasting_data_provider
        expected_train_call = 1 if float(train_ratio_in_tv) == 1.0 else 2
        provider_call_count = 0

        def _provider_with_few_shot(
            data,
            config,
            timeenc,
            batch_size,
            shuffle,
            drop_last,
            sample_timestamp,
            model_name,
        ):
            nonlocal provider_call_count
            provider_call_count += 1
            dataset, loader = original_provider(
                data,
                config,
                timeenc,
                batch_size,
                shuffle,
                drop_last,
                sample_timestamp,
                model_name,
            )

            if provider_call_count == expected_train_call:
                loader = apply_few_shot_loader(
                    dataset=dataset,
                    loader=loader,
                    config=config,
                    drop_last=drop_last,
                    collate_fn=loader.collate_fn,
                )
            return dataset, loader

        deep_base_module.forecasting_data_provider = _provider_with_few_shot
        try:
            return super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
        finally:
            deep_base_module.forecasting_data_provider = original_provider

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ):
        return super().forecast(
            horizon,
            series,
            covariates=self._without_exog_covariates(covariates),
        )

    def batch_forecast(
        self,
        horizon: int,
        batch_maker,
        exog_futures,
        i,
        series_number,
        **kwargs,
    ):
        return super().batch_forecast(
            horizon,
            self._TargetOnlyBatchMaker(batch_maker),
            None,
            i,
            series_number,
            **kwargs,
        )


class BaseSTFMAdapter(STFMAdapterWorkflowBase, STFMDeepForecastingModelBase):
    def __init__(self, model_name, model_class, default_hyper_params, **kwargs):
        super().__init__(default_hyper_params, **kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.check_point = None

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        return self.model_class(self.config)

    def _uses_internal_train_loss(self) -> bool:
        return False

    def _init_criterion(self):
        if self._uses_internal_train_loss():
            return _ZeroLoss()
        return super()._init_criterion()

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del exog_future

        if self._uses_internal_train_loss() and hasattr(self.model, "forward_train"):
            pred, train_loss = self.model.forward_train(input, target)
            return {"output": pred, "additional_loss": train_loss}

        dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float().to(
            input.device
        )
        output = self.model(input, input_mark, dec_input, target_mark)
        if isinstance(output, tuple):
            output = output[0]
        return {"output": output}

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        mode = self._resolve_shot_mode()
        self._setup_post_norm_shot(mode)
        return super().forecast_fit(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )


class STFMAdapter(BaseSTFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, GENERIC_STFM_HPARAMS, **kwargs)


def generate_model_factory(
    model_name: str,
    model_class: type,
    adapter_cls: type,
    required_args: dict,
) -> Dict:
    def model_factory(**kwargs):
        return adapter_cls(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def stfm_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=STFMAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )
