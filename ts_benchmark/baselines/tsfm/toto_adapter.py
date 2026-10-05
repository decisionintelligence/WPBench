import os
import random
import warnings
from typing import Optional, Type

import numpy as np
import pandas as pd
import torch
from einops import rearrange

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    GENERIC_TSFM_HPARAMS,
    TOTO_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.tsfm.toto_official_trainer_helpers import (
    run_toto_official_trainer,
)
from ts_benchmark.baselines.tsfm import toto_trainer_helpers as _toto_tr
from ts_benchmark.baselines.tsfm.models.toto import Toto
from ts_benchmark.baselines.tsfm.submodules.toto.forecaster import (
    TotoForecaster,
    resolve_num_samples_and_batch,
)
from ts_benchmark.baselines.tsfm.toto_helpers import (
    bench_freq_to_seconds,
    build_masked_timeseries,
)
from ts_benchmark.utils.data_processing import infer_series_number


class TotoAdapter(BaseTSFMAdapter):
    @staticmethod
    def _is_missing_value(value) -> bool:
        return value is None or (isinstance(value, str) and value.strip() in {"", "None"})

    def __init__(self, model_name, model_class, **kwargs):
        merged = {**GENERIC_TSFM_HPARAMS, **TOTO_HPARAMS}
        super().__init__(model_name, model_class, merged, **kwargs)

        cfg = self.config
        cfg.pred_len = int(getattr(cfg, "pred_len", getattr(cfg, "horizon", 96)))
        cfg.horizon = int(getattr(cfg, "horizon", cfg.pred_len))
        cfg.pred_len = cfg.horizon
        cfg.seq_len = int(self._resolve_lsf_context_length(cfg))
        cfg.label_len = int(getattr(cfg, "label_len", cfg.horizon))
        cfg.freq = str(getattr(cfg, "freq", "h"))
        cfg.finetune_style = str(getattr(cfg, "finetune_style", "benchmark")).lower()
        if cfg.finetune_style not in {"benchmark", "official_trainer"}:
            raise ValueError(f"Unknown Toto finetune_style: {cfg.finetune_style}")
        if "shot_mode" in kwargs:
            cfg.shot_mode = str(kwargs["shot_mode"]).lower()

        self.CovariateFusion = None
        self._toto_mode = str(getattr(cfg, "shot_mode", "full_shot")).lower()
        self._trained_series_num: Optional[int] = None
        self._trained_series_dim: Optional[int] = None
        self._trained_exog_dim: int = 0
        self._forecaster: Optional[TotoForecaster] = None
        self._toto_train_loop_active = False
        self._toto_finetune_loss = _toto_tr.build_finetune_loss(cfg)
        self._toto_external_scaler_enabled = bool(getattr(cfg, "norm", False))

    def _resolve_random_seed(self) -> int:
        raw_seed = getattr(self.config, "seed", 59)
        try:
            return int(raw_seed)
        except (TypeError, ValueError):
            return 59

    def _align_rng_with_raw_toto(self) -> None:
        seed = self._resolve_random_seed()
        self.config.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if bool(getattr(self.config, "deterministic_algorithms", True)):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_lsf_context_length(cfg, fallback: int = 2048) -> int:
        raw_seq_len = getattr(cfg, "seq_len", None)
        if not TotoAdapter._is_missing_value(raw_seq_len):
            return max(1, int(raw_seq_len))

        for key in ("context_lengths", "context-lengths"):
            raw = getattr(cfg, key, None)
            if TotoAdapter._is_missing_value(raw):
                continue
            if isinstance(raw, (list, tuple)):
                if len(raw) == 0:
                    continue
                return max(1, int(raw[0]))
            return max(1, int(raw))

        return max(1, int(fallback))

    def _load_toto(self) -> Toto:
        ckpt = (
            getattr(self.config, "checkpoint_path", None)
            or getattr(self.config, "ckpt_path", None)
            or getattr(self.config, "pretrain_model_path", None)
            or ""
        )
        ckpt = str(ckpt).strip()
        if not ckpt:
            return self.model_class(self.config)
        if os.path.isdir(ckpt) or os.path.isfile(ckpt):
            return Toto.load_from_checkpoint(ckpt, map_location="cpu")
        return Toto.from_pretrained(ckpt)

    def _init_model(self):
        model = self._load_toto()
        num_exogenous_variables = int(
            max(
                0,
                int(getattr(self.config, "input_dim", 0)) - int(getattr(self.config, "output_dim", 0)),
            )
        )
        if self._uses_toto_finetune_branch() and num_exogenous_variables > 0 and model.model.fusion is None:
            model.model.enable_variate_labels()
        if bool(getattr(self.config, "freeze_backbone", False)):
            for p in model.parameters():
                p.requires_grad = False
        return model

    def _uses_internal_train_loss(self) -> bool:
        return self._uses_toto_finetune_branch() and not self._uses_official_trainer()

    def _uses_official_trainer(self) -> bool:
        return (
            self._toto_mode in {"few_shot", "full_shot"}
            and str(getattr(self.config, "finetune_style", "benchmark")).lower() == "official_trainer"
        )

    def _uses_toto_finetune_branch(self) -> bool:
        return self._toto_mode != "zero_shot" and int(getattr(self.config, "num_epochs", 0)) > 0

    @staticmethod
    def _index_to_unix_seconds(index: pd.Index) -> Optional[np.ndarray]:
        if isinstance(index, pd.PeriodIndex):
            index = index.to_timestamp()
        if isinstance(index, pd.DatetimeIndex):
            return (index.asi8 // 1_000_000_000).astype(np.int64)
        if isinstance(index, pd.TimedeltaIndex):
            return (index.asi8 // 1_000_000_000).astype(np.int64)
        return None

    @staticmethod
    def _scale_numeric_index_to_seconds(values: np.ndarray, freq: str | None) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        interval_sec = max(1, int(bench_freq_to_seconds(freq)))
        if arr.size == 0:
            return np.asarray(arr, dtype=np.int64)

        diffs = np.diff(arr, axis=-1).reshape(-1)
        diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)]
        if diffs.size == 0:
            scale = float(interval_sec)
        else:
            step = float(np.median(np.abs(diffs)))
            if np.isclose(step, float(interval_sec), rtol=0.01, atol=1.0):
                scale = 1.0
            elif np.isclose(step, 1.0, rtol=0.0, atol=1e-6):
                scale = float(interval_sec)
            else:
                scale = float(interval_sec) / step

        return np.rint(arr * scale).astype(np.int64)

    @staticmethod
    def _parse_datetime_like_seconds(values: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.datetime64):
            return (arr.astype("datetime64[ns]").astype(np.int64) // 1_000_000_000).astype(np.int64)
        if np.issubdtype(arr.dtype, np.timedelta64):
            return (arr.astype("timedelta64[ns]").astype(np.int64) // 1_000_000_000).astype(np.int64)
        if np.issubdtype(arr.dtype, np.number):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(arr.reshape(-1), errors="coerce")
        if parsed.isna().any():
            return None
        return (parsed.asi8 // 1_000_000_000).reshape(arr.shape).astype(np.int64)

    @staticmethod
    def _parse_numeric_like_seconds(values: np.ndarray, freq: str | None) -> Optional[np.ndarray]:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.datetime64) or np.issubdtype(arr.dtype, np.timedelta64):
            return None
        numeric = pd.to_numeric(arr.reshape(-1), errors="coerce")
        if np.isnan(numeric).any():
            return None
        numeric_arr = np.asarray(numeric, dtype=np.float64).reshape(arr.shape)
        return TotoAdapter._scale_numeric_index_to_seconds(numeric_arr, freq)

    @staticmethod
    def _infer_interval_seconds_from_seconds_array(
        values: np.ndarray,
        fallback_interval_seconds: int,
    ) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 1-D or 2-D values to infer intervals, got {arr.shape}")

        intervals = np.full((arr.shape[0],), max(1, int(fallback_interval_seconds)), dtype=np.int64)
        if arr.shape[1] <= 1:
            return intervals

        diffs = np.diff(arr, axis=1)
        for idx in range(diffs.shape[0]):
            row = diffs[idx]
            row = row[np.isfinite(row) & (np.abs(row) > 0)]
            if row.size == 0:
                continue
            intervals[idx] = max(1, int(np.rint(np.median(np.abs(row)))))
        return intervals

    @classmethod
    def _index_to_time_features(
        cls,
        index: pd.Index,
        freq: str | None,
    ) -> tuple[np.ndarray, int]:
        fallback_interval = max(1, int(bench_freq_to_seconds(freq)))
        ts_seconds = cls._index_to_unix_seconds(index)
        if ts_seconds is None:
            ts_seconds = cls._parse_numeric_like_seconds(np.asarray(index), freq)
        if ts_seconds is None:
            ts_seconds = cls._parse_datetime_like_seconds(np.asarray(index))
        if ts_seconds is None:
            ts_seconds = np.arange(len(index), dtype=np.int64) * fallback_interval

        inferred_interval = int(cls._infer_interval_seconds_from_seconds_array(ts_seconds, fallback_interval)[0])
        return np.asarray(ts_seconds, dtype=np.int64), inferred_interval

    @staticmethod
    def _batch_label_windows_to_seconds(values: np.ndarray, freq: str | None) -> Optional[np.ndarray]:
        arr = np.asarray(values)
        if arr.ndim != 2:
            return None

        flat = arr.reshape(-1)
        positions, _ = pd.factorize(flat, sort=False)
        if (positions < 0).any():
            return None

        interval_sec = max(1, int(bench_freq_to_seconds(freq)))
        return positions.reshape(arr.shape).astype(np.int64) * interval_sec

    @classmethod
    def _index_to_aligned_seconds(cls, index: pd.Index, freq: str | None) -> np.ndarray:
        ts_seconds, _ = cls._index_to_time_features(index, freq)
        return ts_seconds

    @classmethod
    def _batch_time_features_from_raw_timestamps(
        cls,
        raw_timestamps,
        history_len: int,
        series_number: int,
        freq: str | None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if raw_timestamps is None:
            return None, None

        ts_arr = np.asarray(raw_timestamps)
        if ts_arr.ndim != 2 or ts_arr.shape[1] != history_len:
            return None, None

        fallback_interval = max(1, int(bench_freq_to_seconds(freq)))

        ts_base = cls._parse_numeric_like_seconds(ts_arr, freq)
        if ts_base is None:
            ts_base = cls._parse_datetime_like_seconds(ts_arr)
        if ts_base is None:
            ts_base = cls._batch_label_windows_to_seconds(ts_arr, freq)
        if ts_base is None:
            return None, None

        interval_b = cls._infer_interval_seconds_from_seconds_array(ts_base, fallback_interval)
        last_observed_b = np.repeat(ts_base[:, -1], repeats=int(series_number), axis=0)
        interval_bn = np.repeat(interval_b, repeats=int(series_number), axis=0)
        return last_observed_b.astype(np.int64), interval_bn.astype(np.int64)

    @classmethod
    def _batch_last_observed_from_raw_timestamps(
        cls,
        raw_timestamps,
        history_len: int,
        series_number: int,
        freq: str | None,
    ) -> Optional[np.ndarray]:
        last_observed_b, _ = cls._batch_time_features_from_raw_timestamps(
            raw_timestamps,
            history_len=history_len,
            series_number=series_number,
            freq=freq,
        )
        return last_observed_b

    @staticmethod
    def _extract_last_observed_timestamp_seconds(
        timestamp_seconds_bt: Optional[np.ndarray],
        history_len: int,
    ) -> Optional[np.ndarray]:
        if timestamp_seconds_bt is None:
            return None
        if timestamp_seconds_bt.ndim != 2 or timestamp_seconds_bt.shape[1] != history_len:
            return None
        return np.asarray(timestamp_seconds_bt[:, -1], dtype=np.int64).copy()

    def _move_model_to_inference_device(self) -> torch.device:
        self._assert_model_ready()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        return device

    @staticmethod
    def _flatten_benchmark_tensor(
        tensor: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() == 4:
            return rearrange(tensor, "b n t c -> (b n) t c")
        if tensor.dim() == 3:
            return tensor
        raise ValueError(
            "TotoAdapter expects benchmark tensors shaped [B,T,C] or [B,N,T,C], "
            f"got {tuple(tensor.shape)}."
        )

    def _get_forecaster(self) -> TotoForecaster:
        self._assert_model_ready()
        backbone = self.model.model
        if self._forecaster is None or self._forecaster.model is not backbone:
            self._forecaster = TotoForecaster(backbone)
        return self._forecaster

    def _predict_with_toto_forecaster(
        self,
        input_btc: torch.Tensor,
        prediction_length: int,
        *,
        valid_mask_bt: Optional[torch.Tensor] = None,
        timestamp_seconds_bt: Optional[torch.Tensor] = None,
        last_observed_timestamp_seconds_b: Optional[torch.Tensor] = None,
        time_interval_seconds_b: Optional[torch.Tensor] = None,
        future_exogenous_btc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_btc.dim() != 3:
            raise ValueError(
                f"Toto forecast expects input shaped [B,T,C], got {tuple(input_btc.shape)}."
            )

        patch_stride = int(self.model.model.patch_embed.stride)
        target_dim = int(
            self._trained_series_dim
            if self._trained_series_dim is not None
            else getattr(self.config, "output_dim", input_btc.shape[-1])
        )
        if target_dim > input_btc.shape[-1]:
            raise ValueError(
                "Benchmark input channel count is smaller than Toto target_dim: "
                f"input={input_btc.shape[-1]}, target_dim={target_dim}."
            )
        num_exogenous_variables = int(input_btc.shape[-1] - target_dim)

        masked = build_masked_timeseries(
            input_btc,
            freq=str(getattr(self.config, "freq", "h")),
            patch_stride=patch_stride,
            valid_mask_bt=valid_mask_bt,
            timestamp_seconds_bt=timestamp_seconds_bt,
            last_observed_timestamp_seconds_b=last_observed_timestamp_seconds_b,
            time_interval_seconds_b=time_interval_seconds_b,
        )
        if num_exogenous_variables > 0:
            masked = masked._replace(num_exogenous_variables=num_exogenous_variables)

        future_exogenous_variables = None
        if future_exogenous_btc is not None:
            future_exogenous_btc = self._flatten_benchmark_tensor(future_exogenous_btc)
            if future_exogenous_btc is None:
                raise ValueError("future_exogenous_btc unexpectedly resolved to None.")
            if future_exogenous_btc.dim() != 3:
                raise ValueError(
                    "Toto future exogenous input must be shaped [B,T,C], "
                    f"got {tuple(future_exogenous_btc.shape)}."
                )
            if num_exogenous_variables <= 0:
                raise ValueError(
                    "Benchmark provided future exogenous variables, but Toto input "
                    "does not include matching historical exogenous channels."
                )
            if future_exogenous_btc.shape[-1] != num_exogenous_variables:
                raise ValueError(
                    "Mismatch between Toto historical/future exogenous channel counts: "
                    f"history={num_exogenous_variables}, future={future_exogenous_btc.shape[-1]}."
                )
            future_exogenous_variables = future_exogenous_btc.permute(0, 2, 1).contiguous()

        ns, spb = resolve_num_samples_and_batch(
            getattr(self.config, "num_samples", None),
            int(getattr(self.config, "samples_per_batch", 256)),
        )
        forecast = self._get_forecaster().forecast(
            masked,
            prediction_length=int(prediction_length),
            num_samples=ns,
            samples_per_batch=spb,
            use_kv_cache=bool(getattr(self.config, "use_kv_cache", True)),
            future_exogenous_variables=future_exogenous_variables,
        )
        point_bvt = forecast.median if ns is not None else forecast.mean
        return point_bvt.permute(0, 2, 1).contiguous()

    def _scale_history_btc(self, hist_btc: np.ndarray) -> np.ndarray:
        history = np.asarray(hist_btc, dtype=np.float32)
        if not self._toto_external_scaler_enabled or not bool(getattr(self.config, "norm", False)):
            return history

        batch = history.shape[0]
        series_dim = int(self._trained_series_dim or history.shape[-1])
        exog_dim = int(history.shape[-1] - series_dim)
        if exog_dim > 0:
            scaled_series = self.scaler1.transform(
                rearrange(history[:, :, :series_dim], "b t c -> (b t) c")
            )
            scaled_series = rearrange(scaled_series, "(b t) c -> b t c", b=batch)
            scaled_exog = self.scaler2.transform(
                rearrange(history[:, :, series_dim:], "b t c -> (b t) c")
            )
            scaled_exog = rearrange(scaled_exog, "(b t) c -> b t c", b=batch)
            return np.concatenate((scaled_series, scaled_exog), axis=2)

        scaled_history = self.scaler1.transform(rearrange(history, "b t c -> (b t) c"))
        return rearrange(scaled_history, "(b t) c -> b t c", b=batch)

    def _inverse_scale_prediction_btc(self, pred_btc: np.ndarray) -> np.ndarray:
        prediction = np.asarray(pred_btc, dtype=np.float32)
        if not self._toto_external_scaler_enabled or not bool(getattr(self.config, "norm", False)):
            return prediction

        batch = prediction.shape[0]
        restored = self.scaler1.inverse_transform(
            rearrange(prediction, "b t c -> (b t) c")
        )
        return rearrange(restored, "(b t) c -> b t c", b=batch)

    def _assert_model_ready(self) -> None:
        if self.model is None:
            raise ValueError("Model not ready. Call forecast_fit() first.")

    def _process_with_toto_finetune_loss(self, input, target):
        self._assert_model_ready()
        patch_size = int(self.model.model.patch_embed.patch_size)
        finetune_batch = _toto_tr.prepare_finetune_batch(
            input,
            target,
            horizon=int(self.config.horizon),
            target_dim=int(getattr(self.config, "output_dim", input.shape[-1])),
            patch_size=patch_size,
        )
        output = self.model.model(
            finetune_batch.model_inputs_bvt,
            finetune_batch.input_padding_mask_bvt,
            finetune_batch.id_mask_bvt,
            num_exogenous_variables=finetune_batch.num_exogenous_variables,
        )
        prediction_mask_length = None
        if not self.model.training:
            prediction_mask_length = int(
                getattr(
                    self.config,
                    "toto_val_prediction_len",
                    getattr(self.config, "horizon", patch_size),
                )
            )
        additional_loss = _toto_tr.compute_finetune_loss(
            self._toto_finetune_loss,
            output.distribution,
            output.loc,
            output.scale,
            finetune_batch,
            prediction_mask_length=prediction_mask_length,
        )
        point_output_bvt = _toto_tr.distribution_mean_to_bvt(
            output.distribution,
            output.loc,
            output.scale,
            output_length=finetune_batch.original_input_length,
        )[:, : finetune_batch.target_dim, :]
        return {
            "output": point_output_bvt.permute(0, 2, 1).contiguous(),
            "additional_loss": additional_loss,
        }

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        if self._toto_train_loop_active and self._uses_toto_finetune_branch():
            del input_mark, target_mark, exog_future
            return self._process_with_toto_finetune_loss(input, target)

        del target, input_mark, target_mark

        input_btc = self._flatten_benchmark_tensor(input)
        future_exogenous_btc = self._flatten_benchmark_tensor(exog_future)
        output = self._predict_with_toto_forecaster(
            input_btc,
            int(self.config.horizon),
            future_exogenous_btc=future_exogenous_btc,
        )
        # [B, horizon, C]
        return {"output": output}

    def _reshape_history(
        self,
        target_history: np.ndarray,
        series_number: int,
    ) -> tuple[np.ndarray, int]:
        if target_history.ndim != 3:
            raise ValueError(f"Expected target history shape [B,T,C], got {target_history.shape}")

        _, _, target_total = target_history.shape
        if target_total % series_number != 0:
            raise ValueError("Target columns cannot be evenly divided by series_number.")

        series_dim = target_total // series_number
        hist = rearrange(target_history, "b t (n c) -> (b n) t c", n=series_number, c=series_dim)
        return hist, series_dim

    def _build_context_window(self, hist_btc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if hist_btc.ndim != 3:
            raise ValueError(f"Expected history shape [B,T,C], got {hist_btc.shape}")

        history = np.asarray(hist_btc, dtype=np.float32)
        b, t, c = history.shape
        context_len = max(1, int(getattr(self.config, "seq_len", t)))

        if t >= context_len:
            return history[:, -context_len:, :], np.ones((b, context_len), dtype=bool)

        pad_left = context_len - t
        context = np.zeros((b, context_len, c), dtype=np.float32)
        context[:, pad_left:, :] = history
        observed = np.zeros((b, context_len), dtype=bool)
        observed[:, pad_left:] = True
        return context, observed

    def _predict_with_official_forecaster(
        self,
        hist_btc: np.ndarray,
        horizon: int,
        last_observed_timestamp_seconds_b: Optional[np.ndarray] = None,
        time_interval_seconds_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._assert_model_ready()

        device = self.model.device
        context_btc, observed_bt = self._build_context_window(hist_btc)

        input_btc = torch.as_tensor(context_btc, dtype=torch.float32, device=device)
        observed_mask = torch.as_tensor(observed_bt, dtype=torch.bool, device=device)
        last_observed_tensor = None
        if last_observed_timestamp_seconds_b is not None:
            last_observed_tensor = torch.as_tensor(
                last_observed_timestamp_seconds_b,
                dtype=torch.int64,
                device=device,
            )
        time_interval_tensor = None
        if time_interval_seconds_b is not None:
            time_interval_tensor = torch.as_tensor(
                time_interval_seconds_b,
                dtype=torch.int64,
                device=device,
            )
        point_btc = self._predict_with_toto_forecaster(
            input_btc,
            int(horizon),
            valid_mask_bt=observed_mask,
            last_observed_timestamp_seconds_b=last_observed_tensor,
            time_interval_seconds_b=time_interval_tensor,
        )
        return point_btc.cpu().numpy()

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        if str(getattr(self.config, "shot_mode", "full_shot")).lower() == "zero_shot":
            _toto_tr.ensure_zero_shot_mode(self.config)
        self._toto_mode = str(getattr(self.config, "shot_mode", "full_shot")).lower()
        self._align_rng_with_raw_toto()
        self._toto_finetune_loss = _toto_tr.build_finetune_loss(self.config)
        self._toto_external_scaler_enabled = bool(getattr(self.config, "norm", False))
        if self._uses_official_trainer():
            self.model = self._init_model()
            self.CovariateFusion = None
            self.check_point = None
            result = run_toto_official_trainer(
                self,
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
            self._trained_series_num = int(self.config.series_num)
            self._trained_series_dim = int(self.config.series_dim)
            self._trained_exog_dim = int(self.config.input_dim - self.config.output_dim)
            self._forecaster = None
            self.model.eval()
            self.check_point = self.save_checkpoint({"Model": self.model})
            print(
                "Toto official_trainer finished: "
                f"train_windows={result.train_windows}, "
                f"val_windows={result.val_windows}, "
                f"best_val_loss={result.best_val_loss}, "
                f"best_ckpt_path={result.best_ckpt_path}, "
                f"run_dir={result.run_dir}"
            )
            return self
        self._toto_train_loop_active = self._uses_toto_finetune_branch()
        original_provider = deep_base_module.forecasting_data_provider
        expected_train_call = 1 if float(train_ratio_in_tv) == 1.0 else 2
        provider_call_count = 0

        def _provider_with_toto_training_sampling(
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
            if self._toto_train_loop_active and provider_call_count == expected_train_call:
                loader = _toto_tr.apply_toto_training_loader(
                    dataset,
                    loader,
                    config,
                    drop_last=drop_last,
                    collate_fn=loader.collate_fn,
                )
            return dataset, loader

        if self._toto_train_loop_active:
            deep_base_module.forecasting_data_provider = _provider_with_toto_training_sampling
        try:
            result = super(TotoAdapter, self).forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
        finally:
            deep_base_module.forecasting_data_provider = original_provider
            self._toto_train_loop_active = False
        self._trained_series_num = int(self.config.series_num)
        self._trained_series_dim = int(self.config.series_dim)
        self._trained_exog_dim = int(self.config.input_dim - self.config.output_dim)
        self._forecaster = None
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.eval()
        return result

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        self._assert_model_ready()
        series_number = infer_series_number(series)

        if self._trained_series_num is not None and int(self._trained_series_num) != int(series_number):
            raise ValueError(
                f"series_number mismatch between fit and forecast: fit={self._trained_series_num}, forecast={series_number}."
            )

        target_hist = series.to_numpy(dtype=np.float32, copy=True)[None, :, :]
        hist_btc, series_dim = self._reshape_history(target_hist, series_number)
        ts_1d, interval_sec = self._index_to_time_features(series.index, str(getattr(self.config, "freq", "h")))
        last_observed_ts_b = np.full((hist_btc.shape[0],), ts_1d[-1], dtype=np.int64)
        time_interval_seconds_b = np.full((hist_btc.shape[0],), interval_sec, dtype=np.int64)

        if self._trained_series_dim is not None and int(self._trained_series_dim) != int(series_dim):
            raise ValueError(
                f"series_dim mismatch between fit and forecast: fit={self._trained_series_dim}, forecast={series_dim}."
            )

        if self._trained_exog_dim > 0:
            raise ValueError(
                "Toto forecast currently requires benchmark rolling inference when exogenous channels are enabled."
            )

        hist_btc = self._scale_history_btc(hist_btc)
        pred_btc = self._predict_with_official_forecaster(
            hist_btc,
            int(horizon),
            last_observed_timestamp_seconds_b=last_observed_ts_b,
            time_interval_seconds_b=time_interval_seconds_b,
        )
        pred_btc = self._inverse_scale_prediction_btc(pred_btc)

        pred_wide = rearrange(pred_btc, "(b n) h c -> b h (n c)", b=1, n=series_number, c=series_dim)
        return pred_wide[0]

    def batch_forecast(self, horizon, batch_maker, exog_futures, i, series_number, **kwargs):
        if self._trained_series_num is not None and int(self._trained_series_num) != int(series_number):
            raise ValueError(
                f"series_number mismatch between fit and batch_forecast: fit={self._trained_series_num}, batch={series_number}."
            )
        return super().batch_forecast(horizon, batch_maker, exog_futures, i, series_number, **kwargs)



def toto_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=TotoAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )



def tsfm_adapter(model_info: Type[object]) -> object:
    return toto_adapter(model_info)
