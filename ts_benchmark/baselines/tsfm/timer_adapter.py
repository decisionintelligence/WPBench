from typing import Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    TIMER_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.tsfm.timer_helpers import (
    build_timer_ci_loaders,
    ensure_datetime_index,
    fit_wide_external_scaler,
    inverse_transform_wide_2d,
    inverse_transform_wide_3d,
    infer_timer_freq,
    resolve_external_scaler_mode,
    resolve_timer_data_mode,
    transform_wide_2d,
    transform_wide_3d,
)
from ts_benchmark.baselines.tsfm.few_shot_utils import apply_few_shot_loader
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import infer_series_number


class TimerAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, TIMER_HPARAMS, **kwargs)

        if hasattr(self.config, "finetune_epochs") and int(getattr(self.config, "finetune_epochs", 0)) > 0:
            self.config.num_epochs = int(self.config.finetune_epochs)

        if not hasattr(self.config, "horizon"):
            self.config.horizon = int(getattr(self.config, "pred_len", 0))
        self.config.pred_len = int(getattr(self.config, "pred_len", self.config.horizon))
        self.config.horizon = int(self.config.pred_len)
        self.config.is_spatial = False
        self.config.timer_data_mode = resolve_timer_data_mode(
            getattr(self.config, "timer_data_mode", "official_ci")
        )
        self.config.external_scaler_mode = resolve_external_scaler_mode(
            getattr(self.config, "external_scaler_mode", "pooled_target")
        )

        if int(getattr(self.config, "label_len", -1)) <= 0:
            if getattr(self.config, "use_ims", True):
                self.config.label_len = max(int(self.config.seq_len) - int(self.config.pred_len), 0)
            else:
                self.config.label_len = int(self.config.pred_len)
        else:
            self.config.label_len = int(getattr(self.config, "label_len", 0))

        self.CovariateFusion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.freq = self.config.freq
        self._timer_total_vars = None
        self._timer_original_series_num = None
        self._timer_original_series_dim = None

    @property
    def model_name(self):
        return self._model_name

    def _timer_data_mode(self) -> str:
        mode = resolve_timer_data_mode(getattr(self.config, "timer_data_mode", "official_ci"))
        self.config.timer_data_mode = mode
        return mode

    def _use_official_ci(self) -> bool:
        return self._timer_data_mode() == "official_ci"

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del exog_future

        dec_input = torch.zeros_like(target[:, -self.config.pred_len :, :]).float().to(input.device)
        dec_input = torch.cat([target[:, : self.config.label_len, :], dec_input], dim=1).float()

        output = self.model(input, input_mark, dec_input, target_mark)
        if isinstance(output, tuple):
            output = output[0]
        return {"output": output}

    def _slice_timer_main_loss(self, output, target, series_dim, stage: str):
        if output.ndim == 3:
            out_s = output[:, :, :series_dim]
            tgt_s = target[:, :, :series_dim]
            if bool(self.config.use_ims) and stage in {"train", "vali"}:
                pred = out_s[:, -self.config.seq_len:, :]
                true = tgt_s
            else:
                pred = out_s[:, -self.config.pred_len:, :]
                true = tgt_s[:, -self.config.pred_len:, :]
            return pred, true

        if output.ndim == 4:
            out_s = output[:, :, :, :series_dim]
            tgt_s = target[:, :, :, :series_dim]
            if bool(self.config.use_ims) and stage in {"train", "vali"}:
                pred = out_s[:, :, -self.config.seq_len:, :]
                true = tgt_s
            else:
                pred = out_s[:, :, -self.config.pred_len:, :]
                true = tgt_s[:, :, -self.config.pred_len:, :]
            batch_size, num_series, seq_len, channel_dim = pred.shape
            return pred.reshape(batch_size * num_series, seq_len, channel_dim), true.reshape(
                batch_size * num_series, seq_len, channel_dim
            )

        raise ValueError(f"Unexpected output ndim={output.ndim}")

    def _validate_timer(self, valid_data_loader, series_dim, criterion):
        total_loss = []
        total_count = []
        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                batch_size = int(input.shape[0])
                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                output = out_loss["output"]
                additional_loss = out_loss.get("additional_loss", 0.0)

                pred_main, true_main = self._slice_timer_main_loss(
                    output, target, series_dim, stage="vali"
                )
                pred_main, true_main = self._post_process(pred_main, true_main)
                main_loss = criterion(pred_main, true_main)

                if not torch.is_tensor(additional_loss):
                    additional_loss = main_loss.new_tensor(float(additional_loss))

                total_loss.append((main_loss + additional_loss).detach().cpu().item())
                total_count.append(batch_size)

        self.model.train()
        return float(np.average(total_loss, weights=total_count)) if total_loss else np.inf

    def _build_timer_loaders_legacy(
        self,
        train_data,
        valid_data,
        sample_train_data,
        sample_valid_data,
        config,
    ):
        train_dataset, train_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            sample_timestamp=sample_train_data,
            model_name="Timer",
        )

        train_loader = apply_few_shot_loader(
            dataset=train_dataset,
            loader=train_loader,
            config=config,
            drop_last=False,
            collate_fn=train_loader.collate_fn,
        )

        valid_loader = None
        if (
            valid_data is not None
            and sample_valid_data is not None
            and len(valid_data) > (config.seq_len + config.pred_len)
        ):
            _, valid_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                sample_timestamp=sample_valid_data,
                model_name="Timer",
            )

        return train_loader, valid_loader

    def _configure_official_ci(
        self,
        train_valid_data: pd.DataFrame,
        *,
        series_num: int,
        series_dim: int,
        adj_mx,
        geo_data,
    ) -> None:
        freq = infer_timer_freq(train_valid_data.index, fallback=str(getattr(self.config, "freq", "h")))
        self.config.freq = freq
        self.freq = freq

        self._timer_total_vars = train_valid_data.shape[1]
        self._timer_original_series_num = series_num
        self._timer_original_series_dim = series_dim

        self.config.num_nodes = series_num
        self.config.series_num = series_num
        self.config.series_dim = series_dim
        self.config.timer_total_vars = self._timer_total_vars
        self.config.timer_original_series_num = series_num
        self.config.timer_original_series_dim = series_dim
        self.config.input_dim = 1
        self.config.output_dim = 1
        self.config.enc_in = 1
        self.config.dec_in = 1
        self.config.c_out = 1
        self.config.adj_mx = adj_mx
        self.config.geo_data = geo_data

    def _prepare_timer_model(self):
        self.model = self._init_model()
        self.CovariateFusion = None
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        print("----------------------------------------------------------", self.model_name)

    def _prepare_model_for_inference(self, device: torch.device) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
        self.model.to(device)
        self.model.eval()

    def _forward_timer_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channel_dim = input_tensor.shape
        target_len = int(self.config.label_len) + int(self.config.pred_len)
        input_mark = torch.zeros(batch_size, seq_len, 1, device=input_tensor.device)
        target = torch.zeros(batch_size, target_len, channel_dim, device=input_tensor.device)
        target_mark = torch.zeros(batch_size, target_len, 1, device=input_tensor.device)

        output = self.model(input_tensor, input_mark, target, target_mark)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def _rollout_timer_ci(self, scaled_history: np.ndarray, horizon: int) -> np.ndarray:
        if scaled_history.shape[0] < int(self.config.seq_len):
            raise ValueError("History length is shorter than seq_len for Timer forecast.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_model_for_inference(device)

        rolling = np.asarray(scaled_history, dtype=np.float32)
        predictions = []
        remaining = int(horizon)
        total_vars = rolling.shape[1]

        with torch.no_grad():
            while remaining > 0:
                input_batch = rearrange(rolling[-self.config.seq_len :, :], "l m -> m l 1")
                input_tensor = torch.tensor(input_batch, dtype=torch.float32, device=device)
                output = self._forward_timer_inference(input_tensor)
                chunk = output[:, -self.config.pred_len :, :].detach().cpu().numpy()
                chunk = rearrange(chunk, "m t 1 -> t m")

                take = min(remaining, self.config.pred_len)
                predictions.append(chunk[:take])
                rolling = np.concatenate([rolling, chunk], axis=0)
                remaining -= take

        if not predictions:
            return np.empty((0, total_vars), dtype=np.float32)
        return np.concatenate(predictions, axis=0)

    def _rollout_timer_ci_batch(self, scaled_batch: np.ndarray, horizon: int) -> np.ndarray:
        if scaled_batch.shape[1] < int(self.config.seq_len):
            raise ValueError("Batch history length is shorter than seq_len for Timer batch_forecast.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_model_for_inference(device)

        rolling = np.asarray(scaled_batch, dtype=np.float32)
        batch_size, _, total_vars = rolling.shape
        predictions = []
        remaining = int(horizon)

        with torch.no_grad():
            while remaining > 0:
                input_batch = rearrange(rolling[:, -self.config.seq_len :, :], "b l m -> (b m) l 1")
                input_tensor = torch.tensor(input_batch, dtype=torch.float32, device=device)
                output = self._forward_timer_inference(input_tensor)
                chunk = output[:, -self.config.pred_len :, :].detach().cpu().numpy()
                chunk = rearrange(chunk, "(b m) t 1 -> b t m", b=batch_size, m=total_vars)

                take = min(remaining, self.config.pred_len)
                predictions.append(chunk[:, :take, :])
                rolling = np.concatenate([rolling, chunk], axis=1)
                remaining -= take

        if not predictions:
            return np.empty((batch_size, 0, total_vars), dtype=np.float32)
        return np.concatenate(predictions, axis=1)

    def _rollout_timer_ci_batch_grouped(self, scaled_batch: np.ndarray, horizon: int, *, series_number: int, series_dim: int) -> np.ndarray:
        if scaled_batch.shape[1] < int(self.config.seq_len):
            raise ValueError("Batch history length is shorter than seq_len for Timer batch_forecast.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_model_for_inference(device)

        rolling = np.asarray(scaled_batch, dtype=np.float32)
        batch_size, _, total_vars = rolling.shape
        rolling = rearrange(rolling, "b t (n c) -> b t n c", n=series_number, c=series_dim)
        channel_predictions = []

        with torch.no_grad():
            for channel_index in range(series_dim):
                channel_rolling = rolling[:, :, :, channel_index : channel_index + 1]
                channel_chunks = []
                remaining = int(horizon)

                while remaining > 0:
                    input_batch = rearrange(channel_rolling[:, -self.config.seq_len :, :, 0], "b l n -> (b n) l 1")
                    input_tensor = torch.tensor(input_batch, dtype=torch.float32, device=device)
                    output = self._forward_timer_inference(input_tensor)
                    chunk = output[:, -self.config.pred_len :, :].detach().cpu().numpy()
                    chunk = rearrange(chunk, "(b n) t 1 -> b t n", b=batch_size, n=series_number)

                    take = min(remaining, self.config.pred_len)
                    channel_chunks.append(chunk[:, :take, :])
                    channel_rolling = np.concatenate([channel_rolling, chunk[:, :take, :, None]], axis=1)
                    remaining -= take

                channel_prediction = np.concatenate(channel_chunks, axis=1)
                channel_predictions.append(channel_prediction[..., None])

        if not channel_predictions:
            return np.empty((batch_size, 0, total_vars), dtype=np.float32)

        prediction = np.concatenate(channel_predictions, axis=-1)
        return rearrange(prediction, "b t n c -> b t (n c)")

    def _forecast_fit_legacy(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        if covariates is None:
            covariates = {}
        series_num = infer_series_number(train_valid_data)

        self.config.num_nodes = series_num
        series_dim = train_valid_data.shape[-1] // series_num
        if series_num * series_dim != train_valid_data.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        sample_train_valid_data = train_valid_data.iloc[:, :series_dim]
        train_valid_data = self.reshape_spatiotemporal(train_valid_data, series_num)[0]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            exog_dim = exog_data.shape[-1] // series_num
            sample_exog_data = exog_data.iloc[:, :exog_dim]
            exog_data = self.reshape_spatiotemporal(exog_data, series_num)[0]
            train_valid_data = np.concatenate([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-2]
        else:
            exog_dim = 0

        sample_train_valid_data = (
            pd.concat([sample_train_valid_data, sample_exog_data], axis=1)
            if exog_data is not None
            else sample_train_valid_data
        )
        if sample_train_valid_data.shape[1] == 1:
            self.single_forecasting_hyper_param_tune(sample_train_valid_data)
        else:
            self.multi_forecasting_hyper_param_tune(sample_train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim
        self.config.adj_mx = adj_mx
        self.config.series_num = series_num
        self.config.geo_data = kwargs.get("geo_data", None)

        criterion = self._init_criterion()
        self._prepare_timer_model()
        config = self.config
        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        sample_train_data, sample_valid_data = train_val_split(
            sample_train_valid_data, train_ratio_in_tv, config.seq_len
        )
        train_data_len = train_data.shape[0]
        valid_data_len = valid_data.shape[0] if valid_data is not None else 0

        if exog_dim > 0:
            self.scaler1.fit(rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c"))
            self.scaler2.fit(rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c"))
            if config.norm:
                scaled_series = self.scaler1.transform(
                    rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c")
                )
                train_series = rearrange(scaled_series, "(l n) c -> l c n", l=train_data_len)
                scaled_exog = self.scaler2.transform(
                    rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c")
                )
                train_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=train_data_len)
                train_data = np.concatenate((train_series, train_exog), axis=1)
        else:
            self.scaler1.fit(rearrange(train_data, "l c n -> (l n) c"))
            if config.norm:
                scaled_data = self.scaler1.transform(rearrange(train_data, "l c n -> (l n) c"))
                train_data = rearrange(scaled_data, "(l n) c -> l c n", l=train_data_len)

        if train_ratio_in_tv != 1 and config.norm:
            if exog_dim > 0:
                scaled_series = self.scaler1.transform(
                    rearrange(valid_data[:, :series_dim, :], "l c n -> (l n) c")
                )
                valid_series = rearrange(scaled_series, "(l n) c -> l c n", l=valid_data_len)
                scaled_exog = self.scaler2.transform(
                    rearrange(valid_data[:, series_dim:, :], "l c n -> (l n) c")
                )
                valid_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=valid_data_len)
                valid_data = np.concatenate((valid_series, valid_exog), axis=1)
            else:
                scaled_data = self.scaler1.transform(rearrange(valid_data, "l c n -> (l n) c"))
                valid_data = rearrange(scaled_data, "(l n) c -> l c n", l=valid_data_len)

        valid_for_loader = valid_data if train_ratio_in_tv != 1 else None
        sample_valid_for_loader = sample_valid_data if train_ratio_in_tv != 1 else None
        self.train_data_loader, valid_data_loader = self._build_timer_loaders_legacy(
            train_data=train_data,
            valid_data=valid_for_loader,
            sample_train_data=sample_train_data,
            sample_valid_data=sample_valid_for_loader,
            config=config,
        )

        optimizer = self._init_optimizer(CovariateFusion=self.CovariateFusion)
        scaler = torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model total trainable parameters:{total_params}")
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            for input, target, input_mark, target_mark in self.train_data_loader:
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                additional_loss = out_loss.get("additional_loss", 0)
                output = out_loss["output"]

                pred_main, true_main = self._slice_timer_main_loss(
                    output, target, series_dim, stage="train"
                )
                output, target = self._post_process(pred_main, true_main)
                total_loss = criterion(output, target) + additional_loss

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1:
                valid_loss = self._validate_timer(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint({"Model": self.model})
                if self.early_stopping.early_stop:
                    break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)
        return self

    def _forecast_fit_official_ci(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        del covariates

        train_valid_data = ensure_datetime_index(train_valid_data)
        series_num = infer_series_number(train_valid_data)
        series_dim = train_valid_data.shape[-1] // series_num
        if series_num * series_dim != train_valid_data.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        self._configure_official_ci(
            train_valid_data,
            series_num=series_num,
            series_dim=series_dim,
            adj_mx=adj_mx,
            geo_data=kwargs.get("geo_data", None),
        )

        criterion = self._init_criterion()
        self._prepare_timer_model()
        config = self.config

        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        self.scaler1, train_data, valid_data = fit_wide_external_scaler(
            train_data,
            valid_data,
            norm=bool(config.norm),
            scaler=self.scaler1,
            mode=str(config.external_scaler_mode),
            series_num=series_num,
            series_dim=series_dim,
        )

        _, self.train_data_loader, valid_data_loader = build_timer_ci_loaders(
            train_data,
            valid_data if train_ratio_in_tv != 1 else None,
            config,
        )

        optimizer = self._init_optimizer(CovariateFusion=None)
        scaler = torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model total trainable parameters:{total_params}")
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            for batch_idx, (input, target, input_mark, target_mark) in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                additional_loss = out_loss.get("additional_loss", 0)
                output = out_loss["output"]


                pred_main, true_main = self._slice_timer_main_loss(
                    output, target, 1, stage="train"
                )
                output, target = self._post_process(pred_main, true_main)
                total_loss = criterion(output, target) + additional_loss
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}, Batch Loss: {total_loss.item():.7f}")

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1 and valid_data_loader is not None:
                valid_loss = self._validate_timer(valid_data_loader, 1, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint({"Model": self.model})
                if self.early_stopping.early_stop:
                    break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)
        return self

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        mode = self._resolve_shot_mode()
        self._setup_post_norm_shot(mode)

        if self._use_official_ci():
            return self._forecast_fit_official_ci(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
        return self._forecast_fit_legacy(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        if not self._use_official_ci():
            return super().forecast(horizon, series, covariates=covariates)

        del covariates
        wide_series = ensure_datetime_index(series)
        if self._timer_total_vars is not None and wide_series.shape[1] != self._timer_total_vars:
            raise ValueError(
                "Forecast input column count does not match the Timer training layout: "
                f"expected {self._timer_total_vars}, got {wide_series.shape[1]}."
            )

        scaled_history = transform_wide_2d(
            wide_series.to_numpy(dtype=np.float32, copy=True),
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._timer_original_series_num),
            series_dim=int(self._timer_original_series_dim),
        )

        scaled_prediction = self._rollout_timer_ci(scaled_history, horizon)
        return inverse_transform_wide_2d(
            scaled_prediction,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._timer_original_series_num),
            series_dim=int(self._timer_original_series_dim),
        )

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        exog_futures,
        i,
        series_number,
        **kwargs,
    ) -> np.ndarray:
        if not self._use_official_ci():
            return super().batch_forecast(horizon, batch_maker, exog_futures, i, series_number, **kwargs)

        del exog_futures, i, kwargs

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = np.asarray(input_data["input"], dtype=np.float32)
        if input_np.shape[-1] % series_number != 0:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        batch_size = input_np.shape[0]
        total_vars = input_np.shape[-1]
        series_dim = total_vars // series_number

        # scaled_batch 在series_dim上归一化完毕之后对应了 batch,seq_len series_num*series_dim的形状，接下来送入模型进行预测，输出的scaled_prediction也是这个形状，最后再inverse_transform回去
        scaled_batch = transform_wide_3d(
            input_np,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=series_dim,
        )

        if series_number > 1:
            scaled_prediction = self._rollout_timer_ci_batch_grouped(
                scaled_batch,
                horizon,
                series_number=series_number,
                series_dim=series_dim,
            )
        else:
            scaled_prediction = self._rollout_timer_ci_batch(scaled_batch, horizon)

        # 输入的数据是b t m*1的形态（这个天然适合我们一维的预测目标呢）
        prediction = inverse_transform_wide_3d(
            scaled_prediction,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=series_dim,
        )

        return rearrange(prediction, "b t (n c) -> (b n) t c", n=series_number, c=series_dim)


def timer_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=TimerAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )


def tsfm_adapter(model_info: Type[object]) -> object:
    return timer_adapter(model_info)
