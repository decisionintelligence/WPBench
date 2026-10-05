import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR as VARModel
from typing import Optional

from ts_benchmark.baselines.traditional_model_utils import (
    WideTableScaler,
    build_full_frame,
    infer_layout,
    make_traditional_config,
    normalize_covariate_exog,
    resolve_internal_scaler_mode,
    select_target_positions,
)
from ts_benchmark.models.model_base import ModelBase


class VAR(ModelBase):
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series prediction.
    """

    UNIVARIATE_ERROR = (
        "VAR requires at least 2 variables for each modeling unit, but got 1. "
        "Please use ARIMA, LinearRegressionModel, or RandomForest for "
        "univariate forecasting."
    )
    MULTI_NODE_UNIVARIATE_ERROR = (
        "VAR does not support multi-node univariate data under per-node modeling, "
        "because each node has only 1 variable. Please use ARIMA, "
        "LinearRegressionModel, or RandomForest."
    )

    def __init__(
        self,
        lags=13,
        norm=True,
        scaler_mode=None,
        internal_scaler_mode=None,
        external_scaler_mode="node_variable",
        series_number=None,
        series_dim=None,
        target_channel=None,
        target_column=None,
        target_idx=None,
        target_columns=None,
    ):
        self.lags = lags
        self.results = None
        self.models = []
        self.layout = None
        self.scaler = None
        self.config = make_traditional_config(
            norm=norm,
            scaler_mode=scaler_mode or internal_scaler_mode or "node_variable",
            internal_scaler_mode=internal_scaler_mode,
            external_scaler_mode=external_scaler_mode,
            series_number=series_number,
            series_dim=series_dim,
            target_channel=target_channel,
            target_column=target_column,
            target_idx=target_idx,
            target_columns=target_columns,
        )

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "VAR"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {}

    def _fit_frame(self, frame: pd.DataFrame):
        if frame.shape[1] < 2:
            raise ValueError(self.UNIVARIATE_ERROR)
        model = VARModel(frame)
        results = model.fit(self.lags)
        return results

    def _forecast_frame(
        self,
        results,
        frame: pd.DataFrame,
        horizon: int,
    ) -> np.ndarray:
        return results.forecast(frame.values, steps=horizon)

    def _raise_for_unsupported_univariate(self, kind: str) -> None:
        if kind == "multi_node_univariate":
            raise ValueError(self.MULTI_NODE_UNIVARIATE_ERROR)
        raise ValueError(self.UNIVARIATE_ERROR)

    def forecast_fit(
        self,
        train_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        del train_ratio_in_tv
        self.models = []
        self.layout = None
        self.results = None

        exog = normalize_covariate_exog(train_data, covariates or kwargs.get("covariates"))
        layout = infer_layout(
            train_data,
            exog,
            self.config,
            series_number=kwargs.get("series_number"),
            target_channel=kwargs.get("target_channel"),
        )
        if layout.kind in {"single_node_univariate", "multi_node_univariate"}:
            self._raise_for_unsupported_univariate(layout.kind)

        full_frame = build_full_frame(train_data, exog, layout)
        self.scaler = WideTableScaler(
            norm=bool(self.config.norm),
            mode=resolve_internal_scaler_mode(self.config),
            series_number=layout.series_number,
            series_dim=layout.series_dim,
        )
        scaled_full_frame = self.scaler.fit_transform(full_frame)

        for group in layout.groups:
            if len(group.input_positions) < 2:
                raise ValueError(self.UNIVARIATE_ERROR)
            group_frame = scaled_full_frame.iloc[:, group.input_positions]
            self.models.append(
                {
                    "results": self._fit_frame(group_frame),
                    "input_positions": group.input_positions,
                    "target_positions": group.target_positions,
                }
            )

        self.layout = layout
        self.results = self.models[0]["results"] if self.models else None

        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :return: An array of predicted results.
        """
        if self.layout is None or self.scaler is None or not self.models:
            raise RuntimeError("VAR must be fitted before forecasting.")

        exog = normalize_covariate_exog(series, covariates or kwargs.get("covariates"))
        full_frame = build_full_frame(series, exog, self.layout)
        scaled_full_frame = self.scaler.transform(full_frame)
        scaled_prediction = np.zeros((horizon, len(self.layout.full_columns)))

        for item in self.models:
            input_positions = item["input_positions"]
            group_frame = scaled_full_frame.iloc[:, input_positions]
            scaled_prediction[:, input_positions] = self._forecast_frame(
                item["results"], group_frame, horizon
            )

        full_prediction = self.scaler.inverse_transform(scaled_prediction)
        return full_prediction[:, select_target_positions(self.layout)]


# TODO: VAR_model is kept for backward compatibility, remove all references to VAR_model in the future
VAR_model = VAR
