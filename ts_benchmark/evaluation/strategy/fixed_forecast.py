# -*- coding: utf-8 -*-
import time
from typing import List, Optional, Any

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.utils.data_processing import expand_target_channel, split_channel


class FixedForecast(ForecastingStrategy):
    """
    Fixed forecast strategy class

    This strategy defines a forecasting task with fixed prediction length.

    The required strategy configs include:

    - horizon (int): The length to predict, i.e. the length of the test series;
    - train_ratio_in_tv (float): The ratio of the training series when performing train-validation split.

    The accepted metrics include all regression metrics.

    The return fields other than the specified metrics are (in order):

    - FieldNames.FILE_NAME: The name of the series;
    - FieldNames.FIT_TIME: The training time;
    - FieldNames.INFERENCE_TIME: The inference time;
    - FieldNames.ACTUAL_DATA: The true test data, encoded as a string.
    - FieldNames.INFERENCE_DATA: The predicted data, encoded as a string.
    - FieldNames.LOG_INFO: Any log returned by the evaluator.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "train_ratio_in_tv",
        "save_true_pred",
        "target_channel",
    ]

    # 针对T,(C*N)维的时空数据，需要响应的扩张target_channel，以便后续的提取和划分
    def _expand_target_channels(
        self,
        series: pd.DataFrame,
        base_channels: Any,
        series_number: int,
    ) -> List[int]:
        return expand_target_channel(base_channels, series.shape[1], series_number)

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
        series_number: int,
        adj_mx: Optional[np.ndarray] = None,
        geo_data: Optional[np.ndarray] = None,
    ) -> List:
        model = model_factory()

        target_channel = self._get_scalar_config_value("target_channel", series_name)
        # 特判，如果series_number>1则需要多个这样的target_channel；例如series_name=2,series的维度是(T,20),那么targetchannel就应该是[0,1,10,11]
        if series_number > 1:
            target_channel = self._expand_target_channels(series, target_channel, series_number)
        horizon = self._get_scalar_config_value("horizon", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = data_len - horizon
        if train_length <= 0:
            raise ValueError("The prediction step exceeds the data length")
        # 没有少掉后horizon的exog_series
        _, exog_series = split_channel(series, target_channel)

        train_valid_data, test_data = split_time(series, train_length)
        hpo_eval_mode = self.strategy_config.get("hpo_eval_mode")
        if hpo_eval_mode not in {None, "val", "test"}:
            raise ValueError(f"Unsupported hpo_eval_mode: {hpo_eval_mode}")

        if hpo_eval_mode == "val":
            train_size = int(len(train_valid_data) * train_ratio_in_tv)
            if train_size <= 0 or train_size >= len(train_valid_data):
                raise ValueError(
                    "Invalid train_ratio_in_tv for val-based HPO eval: "
                    "it must create non-empty train and val splits."
                )
            train_data, val_data = split_time(train_valid_data, train_size)
            if len(val_data) < horizon:
                raise ValueError(
                    f"Validation split length ({len(val_data)}) is shorter than horizon ({horizon})."
                )
            target_train_valid_data, exog_train_valid_data = split_channel(
                train_valid_data, target_channel
            )
            target_forecast_data, _ = split_channel(
                train_data, target_channel
            )
            target_eval_data, _ = split_channel(
                split_time(val_data, horizon)[0], target_channel
            )
            _, exog_train_val_data = split_channel(train_valid_data, target_channel)
            covariates = {"exog": exog_train_valid_data}
            test_covariates = {"exog": exog_train_val_data}
            fit_ratio = train_ratio_in_tv
            norm_factor_data = target_forecast_data
        else:
            target_train_valid_data, exog_train_valid_data = split_channel(
                train_valid_data, target_channel
            )
            target_forecast_data = target_train_valid_data
            target_eval_data, _ = split_channel(test_data, target_channel)
            covariates = {"exog": exog_train_valid_data}
            test_covariates = {"exog": exog_series}
            fit_ratio = train_ratio_in_tv
            norm_factor_data = target_train_valid_data

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=fit_ratio,
            adj_mx=adj_mx,
            geo_data=geo_data,
            series_name=series_name,
        )

        end_fit_time = time.time()
        predicted = model.forecast(
            horizon, target_forecast_data, covariates=test_covariates
        )
        end_inference_time = time.time()

        single_series_results, log_info = self.evaluator.evaluate_with_log(
            target_eval_data.to_numpy(),
            predicted,
            # TODO: add configs to control scaling behavior
            self._get_eval_scaler(target_train_valid_data, fit_ratio),
            target_train_valid_data.values,
            norm_factor=self._get_eval_norm_factor(
                series_name, norm_factor_data, series_number
            ),
        )
        inference_data = pd.DataFrame(
            predicted, columns=target_eval_data.columns, index=target_eval_data.index
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(target_eval_data) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(inference_data) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            end_inference_time - end_fit_time,
            actual_data_encoded,
            inference_data_encoded,
            log_info,
        ]

        return single_series_results

    @staticmethod
    def accepted_metrics():
        return regression_metrics.__all__

    @staticmethod
    def default_metrics():
        return regression_metrics.DEFAULT_METRICS

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
