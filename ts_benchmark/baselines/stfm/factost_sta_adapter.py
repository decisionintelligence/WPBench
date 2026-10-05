from typing import Optional, Type
import torch

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import math

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.stfm.adapters_for_stfm import (
    BaseSTFMAdapter,
    GENERIC_STFM_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.stfm.models.factost_sta_io import (
    build_factost_sta_inputs,
    restore_factost_sta_output,
)
from ts_benchmark.baselines.stfm.models.factost_time_features import (
    build_factost_base_time_slots,
)
from ts_benchmark.baselines.utils import DatasetForTransformer

FACTOST_STA_HPARAMS = {
    **GENERIC_STFM_HPARAMS,
    "is_spatial": True,
    "label_len": 0,
    "norm": True,
    # Upstream exp_factost_sta.sh default profile: tiny_4sta.
    "patch_len": 16,
    "stride": 16,
    "factost_num_token": 3,
    "factost_n_layers": 3,
    "factost_n_heads": 4,
    "factost_d_model": 256,
    "factost_d_ff": 1024,
    "factost_dropout": 0.2,
    "factost_embedding_dim": 32,
    "factost_use_st_metadata": True,
    "factost_use_cpr": True,
    "factost_use_st_filtering": True,
    "factost_filter_matrices": "S_s,S_t,S_d",
    "factost_max_delay_steps": 3,
    "factost_revin": True,
}

def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _factost_revin_norm(x_factost: torch.Tensor, eps: float = 1e-5):
    # x_factost: [B, T, N, C], only normalize value channel.
    value = x_factost[..., 0]
    mean = value.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(
        value.var(dim=1, keepdim=True, unbiased=False) + eps
    ).detach()

    x_norm = x_factost.clone()
    x_norm[..., 0] = (value - mean) / stdev
    return x_norm, mean, stdev


def _factost_revin_denorm(output: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor):
    # Vendored FactoST STA normally returns [B, H, N].
    if output.ndim == 3:
        return output * stdev + mean

    # Defensive path if a future wrapper returns [B, N, H, 1].
    if output.ndim == 4 and output.shape[-1] == 1:
        mean_bn = mean.permute(0, 2, 1).unsqueeze(-1)
        stdev_bn = stdev.permute(0, 2, 1).unsqueeze(-1)
        return output * stdev_bn + mean_bn

    raise ValueError(
        f"FactoST RevIN denorm expects output [B,H,N] or [B,N,H,1], got {tuple(output.shape)}."
    )



class FactoSTSTAAdapter(BaseSTFMAdapter):
    """Benchmark adapter for target-only FactoST STA."""

    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, FACTOST_STA_HPARAMS, **kwargs)
        self.config.is_spatial = True
        self.config.label_len = 0

    def _init_scheduler(self, optimizer, train_data_loader):
        lr_type = str(getattr(self.config, "lr_type", "")).strip().lower()
        if lr_type != "sqrt":
            return super()._init_scheduler(optimizer, train_data_loader)

        steps_per_epoch = len(train_data_loader)
        total_steps = int(getattr(self.config, "num_epochs", 0)) * steps_per_epoch
        final_lr_factor = float(getattr(self.config, "factost_sqrt_final_lr_factor", 0.0))

        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0

            progress = step / total_steps
            if progress > 1.0:
                progress = 1.0

            decay = 1 - math.sqrt(progress)
            return final_lr_factor + (1 - final_lr_factor) * decay

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda,
        )


    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, exog_future

        x_factost, future_temporal_features = build_factost_sta_inputs(
            input,
            input_mark,
            target_mark,
            self.config.horizon,
        )

        use_revin = _as_bool(getattr(self.config, "factost_revin", True))
        if use_revin:
            x_factost, revin_mean, revin_stdev = _factost_revin_norm(x_factost)

        output = self.model(
            x_factost,
            future_temporal_features=future_temporal_features,
        )

        if use_revin:
            output = _factost_revin_denorm(output, revin_mean, revin_stdev)

        return {"output": restore_factost_sta_output(output)}


    @staticmethod
    def _build_padding_timestamps(time_stamps_list, padding_len, freq):
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=freq,
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        return np.concatenate((time_stamps_list, padding_time_stamp), axis=1)

    def _padding_time_stamp_mark(self, time_stamps_list, padding_len):
        whole_time_stamp = self._build_padding_timestamps(
            time_stamps_list,
            padding_len,
            self.config.freq,
        )
        return build_factost_base_time_slots(whole_time_stamp)

    @staticmethod
    def _factost_data_provider(
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
            model_name="FactoST_STA",
        )
        raw_dates = pd.to_datetime(sample_timestamp.reset_index()["date"])
        dataset.data_stamp = build_factost_base_time_slots(raw_dates)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # shuffle=False,
            num_workers=config.num_workers,
            drop_last=drop_last,
            # drop_last=True,
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
        deep_base_module.forecasting_data_provider = self._factost_data_provider
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


def factost_sta_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=FactoSTSTAAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )
