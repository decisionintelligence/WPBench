from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd

from ts_benchmark.models.model_base import BatchMaker, ModelBase


def daily_season_length(freq: str) -> int:
    """Return one-day season length for WPBench/GIFT-style frequency strings."""
    normalized = str(freq).strip().lower().replace(" ", "")
    aliases = {
        "1min": 1440,
        "1t": 1440,
        "5min": 288,
        "5mins": 288,
        "5t": 288,
        "10min": 144,
        "10mins": 144,
        "10t": 144,
        "15min": 96,
        "15mins": 96,
        "15t": 96,
        "1hour": 24,
        "1h": 24,
        "hourly": 24,
        "h": 24,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported frequency for daily season length: {freq!r}")
    return aliases[normalized]


class Naive(ModelBase):
    """Last-value naive forecaster compatible with WPBench rolling forecast."""

    def __init__(self, batch_size: int = 1024):
        self.batch_size = int(batch_size)
        self.config = SimpleNamespace(external_scaler_mode="pooled_target")

    @property
    def model_name(self):
        return "Naive"

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "Naive":
        del train_valid_data, covariates, train_ratio_in_tv, kwargs
        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        del covariates, kwargs
        values = np.asarray(series.values, dtype=float)
        if values.shape[0] == 0:
            raise ValueError("Naive requires at least one historical observation.")
        return np.repeat(values[-1:, :], int(horizon), axis=0)

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        exog_future,
        i,
        series_number,
        **kwargs,
    ) -> np.ndarray:
        del exog_future, i, kwargs
        batch = batch_maker.make_batch(self.batch_size, 1)["input"]
        predictions = self._forecast_array(int(horizon), np.asarray(batch, dtype=float))
        return _to_node_major(predictions, int(series_number))

    def _forecast_array(self, horizon: int, history: np.ndarray) -> np.ndarray:
        return np.repeat(history[:, -1:, :], horizon, axis=1)


class SeasonalNaive(Naive):
    """Seasonal naive forecaster following StatsForecast's last-season repeat."""

    def __init__(
        self,
        season_length: Optional[int] = None,
        batch_size: int = 1024,
        freq: Optional[str] = None,
    ):
        super().__init__(batch_size=batch_size)
        if season_length is None:
            season_length = daily_season_length(freq) if freq is not None else 144
        self.season_length = int(season_length)
        if self.season_length <= 0:
            raise ValueError("season_length must be a positive integer.")

    @property
    def model_name(self):
        return "SeasonalNaive"

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        del covariates, kwargs
        values = np.asarray(series.values, dtype=float)
        if values.shape[0] == 0:
            raise ValueError("SeasonalNaive requires at least one historical observation.")
        if values.shape[0] < self.season_length:
            return np.repeat(values[-1:, :], int(horizon), axis=0)
        last_season = values[-self.season_length :, :]
        repeats = int(np.ceil(int(horizon) / self.season_length))
        return np.tile(last_season, (repeats, 1))[: int(horizon), :]

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        exog_future,
        i,
        series_number,
        **kwargs,
    ) -> np.ndarray:
        del exog_future, i, kwargs
        batch = batch_maker.make_batch(self.batch_size, self.season_length)["input"]
        predictions = self._forecast_array(int(horizon), np.asarray(batch, dtype=float))
        return _to_node_major(predictions, int(series_number))

    def _forecast_array(self, horizon: int, history: np.ndarray) -> np.ndarray:
        if history.shape[1] < self.season_length:
            return super()._forecast_array(horizon, history)
        last_season = history[:, -self.season_length :, :]
        repeats = int(np.ceil(horizon / self.season_length))
        return np.tile(last_season, (1, repeats, 1))[:, :horizon, :]


def _to_node_major(predictions: np.ndarray, series_number: int) -> np.ndarray:
    if series_number <= 1:
        return predictions

    batch_size, horizon, total_dim = predictions.shape
    if total_dim % series_number != 0:
        raise ValueError(
            f"Prediction width {total_dim} is not divisible by series_number {series_number}."
        )
    channel_count = total_dim // series_number
    return (
        predictions.reshape(batch_size, horizon, series_number, channel_count)
        .transpose(0, 2, 1, 3)
        .reshape(batch_size * series_number, horizon, channel_count)
    )
