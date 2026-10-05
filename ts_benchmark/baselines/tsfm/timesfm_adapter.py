from typing import Optional, Type

import numpy as np
import pandas as pd
from einops import rearrange

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import BaseTSFMAdapter, TIMESFM_HPARAMS, generate_model_factory
from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.tsfm.submodules.timesfm.models.finetuning_torch import TimesFMFinetuner, FinetuningConfig
from ts_benchmark.baselines.tsfm.timer_helpers import (
    fit_wide_external_scaler,
    inverse_transform_wide_2d,
    inverse_transform_wide_3d,
    resolve_external_scaler_mode,
    transform_wide_2d,
    transform_wide_3d,
)
from ts_benchmark.baselines.tsfm.timesfm_helpers import infer_timesfm_freq_type, wide_to_timesfm_list, \
    timesfm_forecast_to_wide, batch_wide_to_timesfm_list, batch_forecast_back
from ts_benchmark.baselines.tsfm.timesfm_trainer_helpers import TimesFMBenchmarkDataset
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import infer_series_number


class _TimesFMGroupedWindowFewShotDataset:
    """Window-level few-shot wrapper that preserves grouped dataset semantics."""

    def __init__(self, base_dataset, selected_windows):
        self.base_dataset = base_dataset
        self.selected_windows = [int(w) for w in selected_windows]
        self.series_num = int(getattr(base_dataset, "series_num", 1))
        self.series_dim = int(getattr(base_dataset, "series_dim", 1))
        self.total_windows = len(self.selected_windows)

    def __len__(self):
        return self.total_windows * self.series_dim

    def __getitem__(self, index):
        local_window_index = int(index) // self.series_dim
        channel_index = int(index) % self.series_dim
        base_window_index = self.selected_windows[local_window_index]
        base_index = base_window_index * self.series_dim + channel_index
        return self.base_dataset[base_index]


class TimesFMAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, TIMESFM_HPARAMS, **kwargs)
        self.freq_type = 0
        self.total_vars = None
        self._timesfm_shot_mode = str(
            getattr(self.config, "shot_mode", "zero_shot")
        ).lower()
        self.config.external_scaler_mode = resolve_external_scaler_mode(
            getattr(self.config, "external_scaler_mode", "pooled_target")
        )
        self._series_num = None
        self._series_dim = None

    def _build_dataset(self, data_2d: np.ndarray):
        return TimesFMBenchmarkDataset(
            data_2d,
            int(self.config.seq_len),
            int(self.config.horizon),
            self.freq_type,
            int(self.config.timesfm_input_patch_len),
            series_num=int(self._series_num or 1),
            series_dim=int(self._series_dim or 1),
        )

    def forecast_fit(self, train_valid_data: pd.DataFrame, *, covariates: Optional[dict] = None, train_ratio_in_tv: float = 1.0, adj_mx=None, **kwargs):
        del covariates, adj_mx, kwargs
        mode = self._resolve_shot_mode()
        self._setup_post_norm_shot(mode)

        self.total_vars = train_valid_data.shape[1]
        self._series_num = infer_series_number(train_valid_data)
        self._series_dim = self.total_vars // int(self._series_num)
        if int(self._series_num) * int(self._series_dim) != self.total_vars:
            raise ValueError("Data columns cannot be evenly divided by series_num.")
        self.freq_type = infer_timesfm_freq_type(train_valid_data.index, getattr(self.config, "timesfm_freq", "h"))

        train_data, valid_data = train_val_split(
            train_valid_data, float(train_ratio_in_tv), int(self.config.seq_len)
        )
        # Alignment experiment with the original ETTh1 full-shot script:
        # force train/valid DataFrames through float32 before fitting the scaler.
        # Commented out by default to preserve the benchmark's prior behavior.
        # train_data = pd.DataFrame(
        #     train_data.to_numpy(dtype=np.float32, copy=True),
        #     index=train_data.index,
        #     columns=train_data.columns,
        # )
        # if valid_data is not None:
        #     valid_data = pd.DataFrame(
        #         valid_data.to_numpy(dtype=np.float32, copy=True),
        #         index=valid_data.index,
        #         columns=valid_data.columns,
        #     )
        self.scaler1, train_data, valid_data = fit_wide_external_scaler(
            train_data,
            valid_data,
            norm=bool(self.config.norm),
            scaler=self.scaler1,
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._series_num),
            series_dim=int(self._series_dim),
        )
        train_np = train_data.to_numpy(dtype=np.float32, copy=True)
        valid_np = None if valid_data is None else valid_data.to_numpy(dtype=np.float32, copy=True)

        self.model = self._init_model()
        self._timesfm_shot_mode = mode

        if mode == "zero_shot":
            return self

        train_ds = self._build_dataset(train_np)
        if len(train_ds) == 0:
            raise ValueError(
                "TimesFM training dataset is empty. Increase training length, reduce seq_len, or reduce horizon."
            )
        if mode == "few_shot":
            if int(self._series_num) > 1 and hasattr(train_ds, "total_windows") and hasattr(train_ds, "series_dim"):
                sampled_windows = build_few_shot_subset(
                    list(range(int(getattr(train_ds, "total_windows")))),
                    getattr(self.config, "sampling_rate", getattr(self.config, "few_shot_ratio", 0.1)),
                    strategy=getattr(self.config, "sampling_strategy", "uniform"),
                    seed=getattr(self.config, "seed", None),
                )
                if hasattr(sampled_windows, "indices"):
                    selected_windows = [int(i) for i in sampled_windows.indices]
                else:
                    selected_windows = [int(i) for i in sampled_windows]
                train_ds = _TimesFMGroupedWindowFewShotDataset(train_ds, selected_windows)
            else:
                train_ds = build_few_shot_subset(
                    train_ds,
                    getattr(self.config, "sampling_rate", getattr(self.config, "few_shot_ratio", 0.1)),
                    strategy=getattr(self.config, "sampling_strategy", "uniform"),
                    seed=getattr(self.config, "seed", None),
                )

        val_ds = None
        if valid_np is not None:
            candidate_val_ds = self._build_dataset(valid_np)
            if len(candidate_val_ds) > 0:
                val_ds = candidate_val_ds

        finetuner = TimesFMFinetuner(
            model=self.model.finetune_model,
            config=FinetuningConfig(
                batch_size=int(self.config.batch_size),
                num_epochs=int(self.config.num_epochs),
                learning_rate=float(self.config.lr),
                weight_decay=float(getattr(self.config, "weight_decay", 0.01)),
                patience=(
                    int(getattr(self.config, "patience", 0))
                    if val_ds is not None and int(getattr(self.config, "patience", 0)) > 0
                    else None
                ),
                freq_type=self.freq_type,
                use_quantile_loss=bool(getattr(self.config, "timesfm_use_quantile_loss", False)),
                quantiles=list(getattr(self.config, "timesfm_quantiles", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))),
                use_wandb=False,
            ),
        )
        finetuner.finetune(train_ds, val_ds)
        self.check_point = self.save_checkpoint({"Model": self.model.finetune_model})
        return self

    def _timesfm_predict(self, inputs, freq_list):
        if self._timesfm_shot_mode == "zero_shot":
            return self.model.zero_shot_forecast(inputs, freq_list)
        return self.model.finetune_forecast(inputs, freq_list)

    def forecast(self, horizon: int, series: pd.DataFrame, *, covariates: Optional[dict] = None) -> np.ndarray:
        del covariates
        hist = transform_wide_2d(
            series.to_numpy(dtype=np.float32, copy=True),
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._series_num),
            series_dim=int(self._series_dim),
        )
        inputs = wide_to_timesfm_list(hist)
        point_fcst, _ = self._timesfm_predict(inputs, [self.freq_type] * len(inputs))
        pred = timesfm_forecast_to_wide(point_fcst)[:horizon]
        return inverse_transform_wide_2d(
            pred,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._series_num),
            series_dim=int(self._series_dim),
        )

    def batch_forecast(self, horizon, batch_maker, exog_futures, i, series_number, **kwargs):
        del exog_futures, i, kwargs
        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        x = np.asarray(input_data["input"], dtype=np.float32)
        b, t, total_vars = x.shape
        series_dim = total_vars // series_number
        x = transform_wide_3d(
            x,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=series_dim,
        )

        if series_number > 1:
            # Multi-series mode: forecast each aligned channel independently, then
            # stitch the channel-wise outputs back together after prediction.
            x = rearrange(x, "b t (n c) -> b t n c", n=series_number, c=series_dim)
            channel_predictions = []
            for channel_index in range(series_dim):
                channel_inputs = batch_wide_to_timesfm_list(x[:, :, :, channel_index])
                point_fcst, _ = self._timesfm_predict(
                    channel_inputs,
                    [self.freq_type] * len(channel_inputs),
                )
                channel_pred = batch_forecast_back(point_fcst[:, :horizon], b, series_number)
                channel_predictions.append(channel_pred[..., None])

            pred = np.concatenate(channel_predictions, axis=-1)
            pred = rearrange(pred, "b h n c -> b h (n c)")
        else:
            inputs = batch_wide_to_timesfm_list(x)
            point_fcst, _ = self._timesfm_predict(inputs, [self.freq_type] * len(inputs))
            pred = batch_forecast_back(point_fcst[:, :horizon], b, total_vars)

        pred = inverse_transform_wide_3d(
            pred,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=total_vars // series_number,
        )
        series_dim = total_vars // series_number
        return rearrange(pred, "b h (n c) -> (b n) h c", n=series_number, c=series_dim)

def timesfm_adapter(model_info: Type[object]) -> object:
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=TimesFMAdapter,
        required_args={"seq_len": "input_chunk_length", "horizon": "output_chunk_length"},
    )
