from typing import Optional, Type

import pandas as pd
import torch
from torch.utils.data import DataLoader

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.stfm.adapters_for_stfm import (
    BaseSTFMAdapter,
    GENERIC_STFM_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.stfm.models.factost_utp_io import (
    build_factost_utp_inputs,
    restore_factost_utp_output,
)
from ts_benchmark.baselines.utils import DatasetForTransformer

FACTOST_UTP_FINETUNE_HPARAMS = {
    **GENERIC_STFM_HPARAMS,
    "is_spatial": True,
    "label_len": 0,
    "norm": True,
    # Upstream exp_factost_utp.sh default profile: tiny_4utp.
    "patch_len": 16,
    "stride": 16,
    "factost_n_layers": 3,
    "factost_n_heads": 4,
    "factost_d_model": 256,
    "factost_d_ff": 1024,
    "factost_dropout": 0.2,
    "factost_revin": True,
    "lr_type": "OneCycle",
}

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _factost_utp_revin_norm(x_utp: torch.Tensor, eps: float = 1e-5):
    mean = x_utp.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(
        x_utp.var(dim=1, keepdim=True, unbiased=False) + eps
    ).detach()
    return (x_utp - mean) / stdev, mean, stdev


def _factost_utp_revin_denorm(
    output: torch.Tensor,
    mean: torch.Tensor,
    stdev: torch.Tensor,
):
    return output * stdev + mean


class FactoSTUTPFinetuneAdapter(BaseSTFMAdapter):
    """Benchmark adapter for channel-independent FactoST UTP fine-tune."""

    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, FACTOST_UTP_FINETUNE_HPARAMS, **kwargs)
        self.config.is_spatial = True
        self.config.label_len = 0

    def _init_scheduler(self, optimizer, train_data_loader):
        lr_type = str(getattr(self.config, "lr_type", "OneCycle")).strip().lower()
        if lr_type != "onecycle":
            return super()._init_scheduler(optimizer, train_data_loader)

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=float(getattr(self.config, "lr", 1e-4)),
            epochs=int(getattr(self.config, "num_epochs", 1)),
            steps_per_epoch=len(train_data_loader),
            pct_start=float(getattr(self.config, "factost_onecycle_pct_start", 0.3)),
            anneal_strategy=str(
                getattr(self.config, "factost_onecycle_anneal_strategy", "cos")
            ),
            cycle_momentum=_as_bool(
                getattr(self.config, "factost_onecycle_cycle_momentum", True)
            ),
            base_momentum=float(
                getattr(self.config, "factost_onecycle_base_momentum", 0.85)
            ),
            max_momentum=float(
                getattr(self.config, "factost_onecycle_max_momentum", 0.95)
            ),
            div_factor=float(getattr(self.config, "factost_onecycle_div_factor", 25.0)),
            final_div_factor=float(
                getattr(self.config, "factost_onecycle_final_div_factor", 1e4)
            ),
            three_phase=_as_bool(
                getattr(self.config, "factost_onecycle_three_phase", False)
            ),
            last_epoch=int(getattr(self.config, "factost_onecycle_last_epoch", -1)),
        )

    @staticmethod
    def _factost_utp_data_provider(
        data,
        config,
        timeenc,
        batch_size,
        shuffle,
        drop_last,
        sample_timestamp,
        model_name,
    ):
        del model_name
        dataset = DatasetForTransformer(
            dataset=data,
            sample_timestamp=sample_timestamp,
            history_len=config.seq_len,
            prediction_len=config.pred_len,
            label_len=config.label_len,
            timeenc=timeenc,
            freq=config.freq,
            model_name="FactoST_UTP_Finetune",
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=shuffle,
            shuffle=shuffle,
            num_workers=config.num_workers,
            # drop_last=drop_last,
            drop_last=drop_last,
            collate_fn=None,
        )
        return dataset, loader

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        self.config.is_spatial = True
        self.config.label_len = 0

        original_provider = deep_base_module.forecasting_data_provider
        expected_train_call = 1 if float(train_ratio_in_tv) == 1.0 else 2
        provider_call_count = 0

        def _provider_with_factost_utp_loader_alignment(
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

            # Upstream UTP fine-tune uses DataLoaders(..., shuffle_train=True, shuffle_val=False)
            # and its underlying DataLoader leaves drop_last at the default False for both splits.
            is_train_call = provider_call_count == expected_train_call
            aligned_shuffle = True if is_train_call else False
            aligned_drop_last = False

            return self._factost_utp_data_provider(
                data,
                config,
                timeenc,
                batch_size,
                aligned_shuffle,
                aligned_drop_last,
                sample_timestamp,
                model_name,
            )

        deep_base_module.forecasting_data_provider = (
            _provider_with_factost_utp_loader_alignment
        )
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

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark, exog_future
        x_utp, layout = build_factost_utp_inputs(input)

        use_revin = _as_bool(getattr(self.config, "factost_revin", True))
        if use_revin:
            x_utp, revin_mean, revin_stdev = _factost_utp_revin_norm(x_utp)

        output = self.model(x_utp)

        if use_revin:
            output = _factost_utp_revin_denorm(output, revin_mean, revin_stdev)

        return {"output": restore_factost_utp_output(output, layout)}


def factost_utp_finetune_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=FactoSTUTPFinetuneAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )
