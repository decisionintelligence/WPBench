# -*- coding: utf-8 -*-
import itertools
import time
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pandas as pd
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import expand_target_channel, split_channel
from ts_benchmark.utils.data_processing import split_time


class RollingForecastEvalBatchMaker:
    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
        covariates: Optional[dict] = None,
    ):
        self.series = series
        self.index_list = index_list
        self.current_sample_count = 0
        self.covariates = covariates

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data with index and column to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: a batch of data and its time stamps.
        """
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        series = self.series.values
        predict_batch = self._make_batch_data(
            series, np.array(index_list) - win_size, win_size
        )

        indexes = self.series.index
        time_stamps_batch = self._make_batch_data(
            indexes, np.array(index_list) - win_size, win_size
        )
        covariates_batch = self._make_batch_covariates(
            np.array(index_list) - win_size, win_size
        )
        self.current_sample_count += len(index_list)
        return {
            "input": predict_batch,
            "time_stamps": time_stamps_batch,
            "covariates": covariates_batch,
        }

    def make_batch_eval(self, horizon: int) -> dict:
        """
        Return all data to be used for batch evaluation.

        :param horizon: The size of horizon.
        :return: All data to be used for batch evaluation.
        """
        series = self.series.values
        test_batch = self._make_batch_data(series, np.array(self.index_list), horizon)
        covariates_batch = self._make_batch_covariates(
            np.array(self.index_list), horizon
        )
        return {
            "target": test_batch,
            "covariates": covariates_batch,
        }

    def _make_batch_covariates(self, index_list: np.ndarray, win_size: int) -> Dict:
        """
        Create a batch of covariates

        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of covariates.
        """
        covariates = {} if self.covariates is None else self.covariates
        covariates_batch = {}
        if covariates.get("exog") is not None:
            covariates_batch["exog"] = self._make_batch_data(
                covariates["exog"], index_list, win_size
            )
        return covariates_batch

    @staticmethod
    def _make_batch_data(
        data: Any, index_list: np.ndarray, win_size: int
    ) -> np.ndarray:
        """
        Create a batch of data

        :param data: Array_like. Array to create the batch.
        :param index_list: An array of starting indices for each window.
        :param win_size: The size of each window.
        :return: A batch of data.
        """
        windows = sliding_window_view(data, window_shape=(win_size, *data.shape[1:]))
        data_batch = windows[index_list]
        data_batch = np.squeeze(data_batch, axis=tuple(range(1, np.ndim(data))))
        return data_batch

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self.current_sample_count < len(self.index_list)


class RollingForecastPredictBatchMaker(BatchMaker):
    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: A batch of data.
        """
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self._batch_maker.has_more_batches()


class RollingForecast(ForecastingStrategy):
    """
    Rolling forecast strategy class

    This strategy defines a forecasting task that fits once on the training set and
    forecasts on the testing set in a rolling window style.

    The required strategy configs include:

    - horizon (int): The length of each prediction;
    - tv_ratio (float): The ratio of the train-validation series when performing
      train-test split;
    - train_ratio_in_tv (float): The ratio of the training series when performing
      train-validation split;
    - stride (int): Rolling stride, i.e. the interval between two windows;
    - num_rollings (int): The maximum number of steps to forecast;

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
        "tv_ratio",
        "train_ratio_in_tv",
        "stride",
        "num_rollings",
        "save_true_pred",
        "target_channel",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        """
        Get the index list of the rolling windows.

        :param train_length: Training data length.
        :param test_length: Test data length.
        :param horizon: Prediction length.
        :param stride: Rolling stride.
        :return: Index list of the rolling windows.
        """
        data_len = train_length + test_length
        index_list = list(range(train_length, data_len - horizon + 1, stride)) + (
            [data_len - horizon] if (test_length - horizon) % stride != 0 else []
        )
        return index_list

    def _get_split_lens(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        tv_ratio: float,
    ) -> Tuple[int, int]:
        """
        Gets the size of the train-validation series and the test series

        :param series: Target series.
        :param meta_info: Meta-information of the target series.
        :param tv_ratio: The ratio of the train-validation series when performing
            train-test split;
        :return: The length of the train-validation series, and the length of the test series.
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError(
                "The length of training or testing data is less than or equal to 0"
            )
        return train_length, test_length

    # 针对T,(C*N)维的时空数据，需要响应的扩张target_channel，以便后续的提取和划分
    def _expand_target_channels(
        self,
        series: pd.DataFrame,
        base_channels: Any,
        series_number: int,
    ) -> List[int]:
        return expand_target_channel(base_channels, series.shape[1], series_number)

    # 选择归一化的模式
    @staticmethod
    def _uses_node_variable_eval_scaler(model: ModelBase) -> bool:
        config = getattr(model, "config", None)
        mode = str(getattr(config, "external_scaler_mode", "pooled_target")).lower()
        return mode in {"node_variable", "legacy_wide"}

    # t (n c) -> (t n) c
    def _build_eval_scaler_frame(
        self,
        target_train_valid_data: pd.DataFrame,
        series_number: int,
        model: ModelBase,
    ) -> pd.DataFrame:
        if self._uses_node_variable_eval_scaler(model):
            return target_train_valid_data

        # Default benchmark metric normalization pools the same variable across nodes.
        scaler_target_train_valid_data = rearrange(
            target_train_valid_data.values, "t (n c) -> (t n) c", n=series_number
        )
        half_index = target_train_valid_data.index[:series_number]
        repeated_index = list(half_index) * (
            scaler_target_train_valid_data.shape[0] // len(half_index)
        )
        columns = target_train_valid_data.columns[: scaler_target_train_valid_data.shape[1]]
        
        # 由于列名变少了，先取出前两个？
        return pd.DataFrame(
            scaler_target_train_valid_data,
            index=repeated_index,
            columns=columns,
        )

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
        """
        The entry function of execution pipeline of forecasting tasks

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model_factory: The factory to create models.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        model = model_factory()
        """
        save_path = "data/forecasting/rel/adj_mx.npy"

        # 假设 self.adj_mx 是 Tensor 或 numpy.ndarray
        if isinstance(adj_mx, np.ndarray):
            torch.save(adj_mx, save_path)
        print(f"✅ adj_mx saved at {save_path}")
        """
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name, series_number, adj_mx=adj_mx, geo_data=geo_data)
        else:
            return self._eval_batch(series, meta_info, model, series_name, series_number, adj_mx=adj_mx, geo_data=geo_data)

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
        series_number: int,
        adj_mx: Optional[np.ndarray] = None,
        geo_data: Optional[np.ndarray] = None,
    ) -> List:
        """
        The sample execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: the name of the target series.
        :param series_number: The number of sub-series contained in the input series.
        :return: The evaluation results.
        """
        target_channel = self._get_scalar_config_value("target_channel", series_name)
        if series_number > 1:
            target_channel = self._expand_target_channels(series, target_channel, series_number)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_time(series, train_length)
        hpo_eval_mode = self.strategy_config.get("hpo_eval_mode")
        if hpo_eval_mode not in {None, "val", "test"}:
            raise ValueError(f"Unsupported hpo_eval_mode: {hpo_eval_mode}")

        fit_source = train_valid_data
        eval_source = test_data
        fit_ratio = train_ratio_in_tv
        eval_start_index = len(fit_source)
        if hpo_eval_mode == "val":
            train_size = int(len(train_valid_data) * train_ratio_in_tv)
            if train_size <= 0 or train_size >= len(train_valid_data):
                raise ValueError(
                    "Invalid train_ratio_in_tv for val-based HPO eval in rolling_forecast."
                )
            _, eval_source = split_time(train_valid_data, train_size)
            fit_source = train_valid_data
            fit_ratio = train_ratio_in_tv
            eval_start_index = train_size

        target_train_valid_data, exog_data = split_channel(
            fit_source, target_channel
        )
        covariates_train = {}
        covariates_train["exog"] = exog_data



        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=fit_ratio,
            series_number=series_number,
            target_channel=target_channel,
            adj_mx=adj_mx,
            geo_data=geo_data,
            series_name=series_name,
        )
        end_fit_time = time.time()

        scaler_target_train_valid_df = self._build_eval_scaler_frame(
            target_train_valid_data, series_number, model
        )
        eval_scaler = self._get_eval_scaler(scaler_target_train_valid_df, fit_ratio)
        norm_factor = self._get_eval_norm_factor(
            series_name, target_train_valid_data, series_number
        )

        eval_len = len(eval_source)
        index_list = self._get_index(eval_start_index, eval_len, horizon, stride)
        rolling_base = (
            fit_source
            if hpo_eval_mode == "val"
            else pd.concat([fit_source, eval_source], axis=0)
        )
        total_inference_time = 0
        all_targets = []
        all_predicts = []
        all_rolling_actual = []
        all_rolling_predict = []
        for i, index in itertools.islice(enumerate(index_list), num_rollings):
            train, rest = split_time(rolling_base, index)
            # split的含义是取出了rest部分的前horizon作为样例
            test, _ = split_channel(split_time(rest, horizon)[0], target_channel)
            target_train, exog_train = split_channel(train, target_channel)
            covariates_forecast = {}
            covariates_forecast["exog"] = exog_train

            start_inference_time = time.time()
            predict = model.forecast(
                horizon,
                target_train,
                covariates=covariates_forecast,
                series_number=series_number,
                target_channel=target_channel,
            )
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            inference_data = pd.DataFrame(
                predict, columns=test.columns, index=test.index
            )

            all_targets.append(test.to_numpy())
            all_predicts.append(predict)
            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )
        targets = np.stack(all_targets, axis=0)
        all_predicts = np.stack(all_predicts, axis=0)
        if series_number > 1:
            targets = rearrange(targets, "b t (n c) -> b n t c", n=series_number)
            all_predicts = rearrange(all_predicts, "b t (n c) -> b n t c", n=series_number)

        single_series_results = self.evaluator.evaluate(
            targets,
            all_predicts,
            eval_scaler,
            scaler_target_train_valid_df.values,
            norm_factor=norm_factor,
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(all_rolling_actual) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(all_rolling_predict) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
        series_number: int,
        adj_mx: Optional[np.ndarray] = None,
        geo_data: Optional[np.ndarray] = None,
    ) -> List:
        """
        The batch execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: The name of the target series.
        :param series_number: The number of sub-series contained in the input series.
        :param adj_mx: The adjacency matrix if exists.
        :param geo_data: Geographic coordinates (2, N) for spatial patching (PatchSTG only).
        :return: The evaluation results.
        """
        target_channel = self._get_scalar_config_value("target_channel", series_name)
       # 特判，如果series_name>1，则需要多个这样的target_channel；例如series_name=2,series的维度是(T,20),那么targetchannel就应该是[0,1,10,11]
        if series_number > 1:
            if target_channel is not None:
                target_channel = self._expand_target_channels(series, target_channel, series_number)
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)

        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_time(series, train_length)
        hpo_eval_mode = self.strategy_config.get("hpo_eval_mode")
        if hpo_eval_mode not in {None, "val", "test"}:
            raise ValueError(f"Unsupported hpo_eval_mode: {hpo_eval_mode}")

        fit_source = train_valid_data
        eval_source = test_data
        fit_ratio = train_ratio_in_tv
        eval_start_index = len(fit_source)
        if hpo_eval_mode == "val":
            train_size = int(len(train_valid_data) * train_ratio_in_tv)
            if train_size <= 0 or train_size >= len(train_valid_data):
                raise ValueError(
                    "Invalid train_ratio_in_tv for val-based HPO eval in rolling_forecast."
                )
            _, eval_source = split_time(train_valid_data, train_size)
            fit_source = train_valid_data
            fit_ratio = train_ratio_in_tv
            eval_start_index = train_size

        target_train_valid_data, exog_train_valid_data = split_channel(
            fit_source, target_channel
        )
        rolling_base = (
            fit_source
            if hpo_eval_mode == "val"
            else pd.concat([fit_source, eval_source], axis=0)
        )
        target4batch, exog_data4batch = split_channel(rolling_base, target_channel)
        covariates_train, covariates4batch = {}, {}
        covariates_train["exog"] = exog_train_valid_data
        covariates4batch["exog"] = exog_data4batch

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data,
            covariates=covariates_train,
            train_ratio_in_tv=fit_ratio,
            series_number=series_number,
            target_channel=target_channel,
            adj_mx=adj_mx,
            geo_data=geo_data,
            series_name=series_name,
        )
        end_fit_time = time.time()
        
        # 如果是node_variable形态，那我们的归一化仅仅是基于含义是：保留原始宽表 (T, N*C)，如果走默认的pooled_target rearrange(target_train_valid_data.values, "t (n c) -> (t n) c", n=series_number)
        scaler_target_train_valid_df = self._build_eval_scaler_frame(
            target_train_valid_data, series_number, model
        )
        eval_scaler = self._get_eval_scaler(scaler_target_train_valid_df, fit_ratio)
        norm_factor = self._get_eval_norm_factor(
            series_name, target_train_valid_data, series_number
        )

        eval_len = len(eval_source)
        index_list = self._get_index(eval_start_index, eval_len, horizon, stride)
        index_list = index_list[:num_rollings]

        batch_maker = RollingForecastEvalBatchMaker(
            target4batch,
            index_list,
            covariates4batch,
        )

        all_predicts = []
        total_inference_time = 0
        predict_batch_maker = RollingForecastPredictBatchMaker(batch_maker)
        exog_futures = batch_maker.make_batch_eval(horizon)["covariates"].get(
            "exog", None
        )
        i = 0
        # 或许需要predicts输出两个版本，然后比较
        while predict_batch_maker.has_more_batches():
            start_inference_time = time.time()
            predicts = model.batch_forecast(
                horizon, predict_batch_maker, exog_futures, i, series_number
            )
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time
            all_predicts.append(predicts)
            i = i + 1
        print("预测数据的total_inference_time:", total_inference_time)
        # 可能需要额外准备一个涉及到四维输出的版本，此处先存疑考量，但是这种方法肯定也需要相关的还原和存储
        all_predicts = np.concatenate(all_predicts, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]
        targets_wide = targets
        all_predicts_wide = all_predicts
        if series_number > 1:
            all_predicts_wide = rearrange(
                all_predicts_wide, "(b n) t c -> b t (n c)", n=series_number
            )
        # 对于 targets, 也需要转换形状，使其满足(b*n) t c
        targets = rearrange(targets, "b t (n c) -> (b n) t c", n=series_number)
        if len(targets) != len(all_predicts):
            raise RuntimeError("Predictions' len don't equal targets' len!")
        # 此时b,n 已经乘在一起

        if series_number>1:
            all_predicts = rearrange(all_predicts, "(b n) t c -> b n t c", n=series_number)
            targets = rearrange(targets, "(b n) t c -> b n t c", n=series_number)
        all_test_results = []
        start_prediction_time = time.time()

        # 初步验证理解了Batch_size维度可能确实存在损耗情况(有待讨论）
        # 我们可能依然需要使用for循环进行操作，然后对于时空的专门提出一个计算比较合理
        # 这个确实应该把batch放在外面，对于B,N,T,C再设计一个针对三维的探究下即可，而不需要额外的考量将B,T,C视作整体
        """
        for predicts, target in zip(all_predicts, targets):
            single_series_results = self.evaluator.evaluate(
                target,
                predicts,
                eval_scaler,
                scaler_target_train_valid_df.values,
            )
            all_test_results.append(single_series_results)
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
        """

        single_series_results = self.evaluator.evaluate(
            targets,
            all_predicts,
            eval_scaler,
            scaler_target_train_valid_df.values,
            norm_factor=norm_factor,
           # norm_factor=None,  # 先不进行归一化，看看结果，如果后续需要再打开
        )

        
        end_prediction_time = time.time()
        print(f"指标计算运行时间{end_prediction_time-start_prediction_time}秒")

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = self._encode_data(targets_wide) if save_true_pred else np.nan
        inference_data_encoded = (
            self._encode_data(all_predicts_wide) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            actual_data_encoded,
            inference_data_encoded,
            "",
        ]
        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
        return regression_metrics.__all__

    @staticmethod
    def default_metrics() -> List[str]:
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
